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


class PineconeEmbeddingModel:
    """
    Uses Pinecone's native inference API (multilingual-e5-large, 1024-dim).
    No HuggingFace dependency. Works on Vercel since Pinecone is already
    reachable (same API we use for index queries).
    """
    def __init__(self):
        self._pc = None
        self._load()

    def _load(self):
        try:
            from pinecone import Pinecone
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                logger.error("PINECONE_API_KEY not set - embeddings will be zero vectors.")
                return
            self._pc = Pinecone(api_key=api_key)
            logger.info("Pinecone inference client ready (multilingual-e5-large).")
        except Exception as e:
            logger.error(f"Pinecone inference init failed: {e}")

    def encode(self, query, show_progress_bar=False):
        if self._pc is None:
            return np.zeros(1024)
        try:
            result = self._pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[query],
                parameters={"input_type": "query", "truncate": "END"}
            )
            return np.array(result[0].values)
        except Exception as e:
            logger.error(f"Pinecone embed failed: {e}")
            return np.zeros(1024)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = PineconeEmbeddingModel()
    return _embedding_model




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


def vector_search(query: str, top_k: int = 5) -> list:
    """
    Vector Retrieval using Neo4j's native Vector Index.
    """
    model = get_embedding_model()
    driver = get_neo4j_driver()

    if driver is None:
        logger.warning("Neo4j driver unavailable. Skipping vector search.")
        return []

    query_embedding = model.encode(query, show_progress_bar=False).tolist()

    retrieved_texts = []
    
    # We query the unified chunk_embedding vector index
    cypher = """
    CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $query_embedding)
    YIELD node, score
    RETURN node.id AS id, node.text AS text, node.chunk_type AS chunk_type, node.metadata AS metadata, score
    """
    
    try:
        with driver.session() as session:
            result = session.run(cypher, top_k=top_k * 2, query_embedding=query_embedding)
            
            for record in result:
                doc = record["text"]
                chunk_type = record["chunk_type"]
                metadata_str = record["metadata"]
                
                # Hierarchical Child-to-Parent Interception
                if chunk_type == "idx_parent_child" and metadata_str and "parent_chunk_id" in metadata_str:
                    try:
                        import ast
                        meta = ast.literal_eval(metadata_str)
                        if "parent_chunk_id" in meta:
                            parent_id = meta["parent_chunk_id"]
                            # Fetch parent node
                            parent_res = session.run("MATCH (p:DocumentChunk {id: $pid}) RETURN p.text AS text", pid=parent_id)
                            parent_record = parent_res.single()
                            if parent_record:
                                doc = parent_record["text"]
                    except Exception as e:
                        logger.error(f"Failed to fetch parent chunk: {e}")
                
                if doc not in retrieved_texts:
                    retrieved_texts.append(doc)
                    if len(retrieved_texts) >= top_k:
                        break
                        
        return retrieved_texts
    except Exception as e:
        logger.error(f"Neo4j vector search failed: {e}")
        return []


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

    vector_texts = vector_search(query)
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
