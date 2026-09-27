import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from neo4j import GraphDatabase
import requests
import numpy as np
import concurrent.futures

from src.query_router import route_query

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Singleton resource managers
embedding_model = None
pinecone_client = None
pinecone_index = None
neo4j_driver = None

class HFCloudEmbeddingModel:
    def __init__(self):
        self.api_key = os.getenv("HF_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
    def encode(self, query, show_progress_bar=False):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": query})
        if response.status_code != 200:
            logger.error(f"HF API Error: {response.text}")
            return np.zeros(384) # Fallback empty vector
        return np.array(response.json())

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = HFCloudEmbeddingModel()
    return embedding_model

def get_pinecone_index():
    global pinecone_client, pinecone_index
    if pinecone_index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        pinecone_client = Pinecone(api_key=api_key)
        pinecone_index = pinecone_client.Index(index_name)
    return pinecone_index

def get_neo4j_driver():
    global neo4j_driver
    if neo4j_driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    return neo4j_driver

def query_single_collection(index, namespace: str, query_embedding: list, top_k: int = 5) -> list:
    """Helper to query a single Pinecone namespace and return ranked documents."""
    try:
        # Ensure we only fetch child nodes for parent_child index, regular nodes for others
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
    
    query_embedding = model.encode(query, show_progress_bar=False).tolist()
    namespaces = ['idx_fixed', 'idx_recursive', 'idx_semantic', 'idx_parent_child']
    
    all_results = []
    
    # Concurrently query all 4 isolated Pinecone namespaces
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(query_single_collection, index, ns, query_embedding, top_k): ns 
            for ns in namespaces
        }
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())
            
    # Reciprocal Rank Fusion (RRF) Re-ranking Algorithm
    rrf_scores = {}
    
    for item in all_results:
        text = item["text"].strip()
        if not text:
            continue
            
        rank = item["rank"]
        # RRF mathematical scoring formula
        score = 1.0 / (60.0 + rank)
        
        # Aggregate scores for identical text segments
        if text in rrf_scores:
            rrf_scores[text] += score
        else:
            rrf_scores[text] = score
            
    # Sort globally by final RRF scores
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Isolate the top 3 unique highest-scoring winning text matrices
    top_3_texts = [text for text, score in sorted_rrf[:3]]
    return top_3_texts

def graph_search(entities_dict: dict) -> list:
    """
    Parallel Graph Traversal Path leveraging extracted entities.
    """
    driver = get_neo4j_driver()
    
    # Flatten extracted entities
    entities = []
    for category in entities_dict.values():
        entities.extend(category)
        
    if not entities:
        return []
        
    facts = []
    
    # Multi-hop Cypher traversal query pulling neighboring entity nodes up to 2 hops away
    query = """
    MATCH (c)
    WHERE ANY(entity IN $entities WHERE toLower(c.name) CONTAINS toLower(entity))
    OPTIONAL MATCH (c)-[r1]->(t1)
    OPTIONAL MATCH (t1)-[r2]->(t2)
    RETURN c.name AS n1, type(r1) AS rel1, t1.name AS n2, type(r2) AS rel2, t2.name AS n3
    LIMIT 50
    """
    
    try:
        with driver.session(database="neo4j") as session:
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
        logger.error(f"Neo4j connection timeout or traversal failure: {e}")
        
    return facts

def generate_hybrid_context(query: str) -> str:
    """
    Context Fusion Output using Hierarchical Ensemble Retrieval (RRF) and Graph.
    """
    # 1. Parallel Vector Path (RRF)
    vector_texts = vector_search_rrf(query)
    
    # 2. Parallel Graph Path
    entities = route_query(query)
    graph_facts = graph_search(entities)
    
    # 3. Stitch Context Fusion Output
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
