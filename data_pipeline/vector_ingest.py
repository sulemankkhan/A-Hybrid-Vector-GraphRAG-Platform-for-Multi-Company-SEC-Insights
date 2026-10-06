import os
import json
import time
import logging
from pathlib import Path
import chromadb
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_to_chroma():
    # Setup local ChromaDB vector store
    db_path = str(Path("data/vector_store").absolute())
    client = chromadb.PersistentClient(path=db_path)
    
    # Mapping of indices to their respective processed JSON files
    collections_config = [
        {"name": "idx_fixed", "file": "fixed_chunks.json"},
        {"name": "idx_recursive", "file": "recursive_chunks.json"},
        {"name": "idx_semantic", "file": "semantic_chunks.json"},
        {"name": "idx_parent_child", "file": "hierarchical_chunks.json"}
    ]
    
    # Load embedding model once for high performance
    model_name = 'all-MiniLM-L6-v2'
    logger.info(f"Loading {model_name}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    
    processed_dir = Path("data/processed")
    batch_size = 100
    
    for config in collections_config:
        coll_name = config["name"]
        file_name = config["file"]
        file_path = processed_dir / file_name
        
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist. Skipping index {coll_name}.")
            continue
            
        logger.info(f"Starting ingestion for {coll_name} from {file_name}")
        start_time = time.time()
        
        # Read the respective JSON chunk file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}")
            continue
            
        if not data:
            logger.info(f"No data found in {file_name}. Skipping index {coll_name}.")
            continue
            
        # Create or get isolated collection
        collection = client.get_or_create_collection(name=coll_name)
        
        # High-performance batch upsert mechanism
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            ids = []
            texts = []
            metadatas = []
            
            for item in batch:
                ids.append(item["chunk_id"])
                texts.append(item["text"])
                
                # Ingest metadata payload
                metadata = {
                    "company": item.get("company", "Unknown"),
                    "strategy": item.get("strategy", "Unknown"),
                    "source_section": item.get("source_section", "Unknown")
                }
                
                # Explicit handling for hierarchical parent-child metadata
                if "is_parent" in item:
                    metadata["is_parent"] = bool(item["is_parent"])
                if item.get("parent_id"):
                    metadata["parent_chunk_id"] = str(item["parent_id"])
                    
                metadatas.append(metadata)
                
            # Embed the text using sentence-transformers
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()
            
            # Register them into the corresponding database index
            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
        elapsed = time.time() - start_time
        logger.info(f"Finished ingesting {len(data)} items to {coll_name} in {elapsed:.2f} seconds.")
        
if __name__ == "__main__":
    ingest_to_chroma()
