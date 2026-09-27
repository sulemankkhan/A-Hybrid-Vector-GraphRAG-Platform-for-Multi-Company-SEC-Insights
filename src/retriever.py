import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from neo4j import GraphDatabase
import torch
from sentence_transformers import SentenceTransformer
import concurrent.futures

from src.query_router import route_query

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Singleton resource managers
embedding_model = None
chroma_client = None
neo4j_driver = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        # Map to Apple Silicon MPS backend if available
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return embedding_model

def get_chroma_client():
    global chroma_client
    if chroma_client is None:
        db_path = str(Path("data/vector_store").absolute())
        chroma_client = chromadb.PersistentClient(path=db_path)
    return chroma_client

def get_neo4j_driver():
    global neo4j_driver
    if neo4j_driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    return neo4j_driver

def query_single_collection(client, collection_name: str, query_embedding: list, top_k: int = 5) -> list:
    """Helper to query a single ChromaDB collection and return ranked documents."""
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Failed to access ChromaDB collection {collection_name}: {e}")
        return []
        
    try:
        # Ensure we only fetch child nodes for parent_child index, regular nodes for others
        where_clause = {"is_parent": False} if collection_name == "idx_parent_child" else None
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        retrieved_docs = []
        for rank, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            # Hierarchical Child-to-Parent Interception
            if collection_name == "idx_parent_child" and meta and "parent_chunk_id" in meta:
                parent_id = meta["parent_chunk_id"]
                try:
                    parent_results = collection.get(ids=[parent_id])
                    if parent_results and parent_results.get("documents"):
                        doc = parent_results["documents"][0]
                except Exception as e:
                    logger.error(f"Failed to fetch parent chunk {parent_id}: {e}")
                    
            retrieved_docs.append({
                "text": doc,
                "rank": rank + 1,
                "source": collection_name
            })
            
        return retrieved_docs
    except Exception as e:
        logger.error(f"Query failed on collection {collection_name}: {e}")
        return []

def vector_search_rrf(query: str, top_k: int = 5) -> list:
    """
    Parallel Vector Retrieval with Reciprocal Rank Fusion (RRF).
    """
    model = get_embedding_model()
    client = get_chroma_client()
    
    query_embedding = model.encode(query, show_progress_bar=False).tolist()
    collections = ['idx_fixed', 'idx_recursive', 'idx_semantic', 'idx_parent_child']
    
    all_results = []
    
    # Concurrently query all 4 isolated ChromaDB collections
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(query_single_collection, client, col, query_embedding, top_k): col 
            for col in collections
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
