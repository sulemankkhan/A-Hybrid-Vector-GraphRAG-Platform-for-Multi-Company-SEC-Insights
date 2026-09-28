import os
import sys
import logging
from dotenv import load_dotenv
import numpy as np
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Deferred singleton references — nothing connects at import time
_pinecone_index = None
_neo4j_driver = None
_embedding_model = None


class HFCloudEmbeddingModel:
    """
    HuggingFace Inference API v2 — matches the 384-dim all-MiniLM-L6-v2
    vectors already stored in Pinecone. Works on Vercel (DNS resolves there).
    Falls back to zero vectors on local Mac where HF DNS is blocked.
    """
    def __init__(self):
        self.api_key = os.getenv("HF_TOKEN", "")
        # v2 API — more reliable, no cold-start delays
        self.api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def encode(self, query, show_progress_bar=False):
        try:
            import requests as req
            response = req.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": query},
                timeout=15
            )
            if response.status_code != 200:
                logger.error(f"HF API Error {response.status_code}: {response.text}")
                return np.zeros(384)
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    return np.array(data[0])
                return np.array(data)
            return np.zeros(384)
        except Exception as e:
            logger.error(f"HF API call failed (DNS/network): {e}")
            return np.zeros(384)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HFCloudEmbeddingModel()
    return _embedding_model


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        try:
            from pinecone import Pinecone
            api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME")
            if not api_key or not index_name:
                logger.error("Pinecone credentials not set.")
                return None
            client = Pinecone(api_key=api_key)
            _pinecone_index = client.Index(index_name)
            logger.info(f"Pinecone index '{index_name}' connected.")
        except Exception as e:
            logger.error(f"Pinecone init failed: {e}")
            return None
    return _pinecone_index


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Neo4j driver connected.")
        except Exception as e:
            logger.error(f"Neo4j init failed: {e}")
            return None
    return _neo4j_driver


def query_single_collection(index, namespace: str, query_embedding: list, top_k: int = 5) -> list:
    """Helper to query a single Pinecone namespace and return ranked documents."""
    try:
        filter_clause = {"is_parent": False} if namespace == "idx_parent_child" else None

        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_clause
        )

        if not results.matches:
            return []

        retrieved_docs = []
        for rank, match in enumerate(results.matches):
            meta = match.metadata
            doc = meta.get("text", "")

            # Hierarchical Child-to-Parent Interception
            if namespace == "idx_parent_child" and meta and "parent_chunk_id" in meta:
                parent_id = meta["parent_chunk_id"]
                try:
                    parent_results = index.fetch(ids=[parent_id], namespace=namespace)
                    if parent_results and parent_id in parent_results.vectors:
                        doc = parent_results.vectors[parent_id].metadata.get("text", doc)
                except Exception as e:
                    logger.error(f"Failed to fetch parent chunk {parent_id}: {e}")

            retrieved_docs.append({
                "text": doc,
                "rank": rank + 1,
                "source": namespace
            })

        return retrieved_docs
    except Exception as e:
        logger.error(f"Query failed on namespace {namespace}: {e}")
        return []


def vector_search_rrf(query: str, top_k: int = 5) -> list:
    """
    Parallel Vector Retrieval with Reciprocal Rank Fusion (RRF).
    """
    model = get_embedding_model()
    index = get_pinecone_index()

    if index is None:
        logger.warning("Pinecone index unavailable. Skipping vector search.")
        return []

    query_embedding = model.encode(query, show_progress_bar=False).tolist()
    namespaces = ['idx_fixed', 'idx_recursive', 'idx_semantic', 'idx_parent_child']

    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(query_single_collection, index, ns, query_embedding, top_k): ns
            for ns in namespaces
        }
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    # Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    for item in all_results:
        text = item["text"].strip()
        if not text:
            continue
        rank = item["rank"]
        score = 1.0 / (60.0 + rank)
        if text in rrf_scores:
            rrf_scores[text] += score
        else:
            rrf_scores[text] = score

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [text for text, score in sorted_rrf[:3]]


def graph_search(entities_dict: dict) -> list:
    """
    Multi-hop Knowledge Graph traversal using Neo4j.
    """
    driver = get_neo4j_driver()

    if driver is None:
        logger.warning("Neo4j driver unavailable. Skipping graph search.")
        return []

    entities = []
    for category in entities_dict.values():
        entities.extend(category)

    if not entities:
        return []

    facts = []

    query = """
    MATCH (c)
    WHERE ANY(entity IN $entities WHERE toLower(c.name) CONTAINS toLower(entity))
    OPTIONAL MATCH (c)-[r1]->(t1)
    OPTIONAL MATCH (t1)-[r2]->(t2)
    RETURN c.name AS n1, type(r1) AS rel1, t1.name AS n2, type(r2) AS rel2, t2.name AS n3
    LIMIT 50
    """

    try:
        with driver.session() as session:  # No database arg — Aura auto-routes
            result = session.run(query, entities=entities)
            for record in result:
                n1, rel1, n2, rel2, n3 = record["n1"], record["rel1"], record["n2"], record["rel2"], record["n3"]
                if n1 and rel1 and n2:
                    fact = f"Fact: {n1} {rel1} {n2}."
                    if fact not in facts:
                        facts.append(fact)
                if n2 and rel2 and n3:
                    fact2 = f"Fact: {n2} {rel2} {n3}."
                    if fact2 not in facts:
                        facts.append(fact2)
    except Exception as e:
        logger.error(f"Neo4j traversal failure: {e}")

    return facts


def generate_hybrid_context(query: str) -> str:
    """
    Context Fusion Output using Hierarchical Ensemble Retrieval (RRF) and Graph.
    """
    from src.query_router import route_query

    vector_texts = vector_search_rrf(query)
    entities = route_query(query)
    graph_facts = graph_search(entities)

    fusion_parts = []

    if graph_facts:
        fusion_parts.append("### Structured Knowledge Graph Facts ###")
        fusion_parts.append("\n".join(graph_facts))
        fusion_parts.append("\n")

    if vector_texts:
        fusion_parts.append("### Relevant Document Excerpts ###")
        fusion_parts.append("\n\n---\n\n".join(vector_texts))

    hybrid_context = "\n".join(fusion_parts)

    if not hybrid_context.strip():
        return "No relevant context found."

    return hybrid_context


if __name__ == "__main__":
    test_q = "What are the structural dependencies between ASML and global foundries like TSMC and Samsung?"
    context = generate_hybrid_context(test_q)
    print(context)
