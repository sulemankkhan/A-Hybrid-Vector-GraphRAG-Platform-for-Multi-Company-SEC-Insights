import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from pinecone import Pinecone
from tqdm import tqdm

def migrate():
    load_dotenv()
    
    pc_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not pc_api_key or not index_name:
        print("Error: Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env")
        return
        
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=pc_api_key)
    index = pc.Index(index_name)
    
    print("Connecting to local ChromaDB...")
    db_path = str(Path("data/vector_store").absolute())
    client = chromadb.PersistentClient(path=db_path)
    
    collections = ['idx_fixed', 'idx_recursive', 'idx_semantic', 'idx_parent_child']
    
    for col_name in collections:
        print(f"\nProcessing Collection: {col_name}")
        try:
            col = client.get_collection(name=col_name)
        except Exception as e:
            print(f"  -> Skipping {col_name}, collection not found.")
            continue
            
        data = col.get(include=['embeddings', 'metadatas', 'documents'])
        ids = data['ids']
        embeddings = data['embeddings']
        metadatas = data['metadatas']
        documents = data['documents']
        
        if not ids:
            print(f"  -> {col_name} is empty.")
            continue
            
        print(f"  -> Found {len(ids)} vectors. Uploading to Pinecone Namespace '{col_name}'...")
        
        vectors_to_upsert = []
        for i in range(len(ids)):
            meta = metadatas[i] if metadatas[i] else {}
            # Pinecone retrieves context via metadata, so we inject the chunk text into it
            meta['text'] = documents[i] 
            
            # Pinecone metadata values cannot be None, filter them
            clean_meta = {k: v for k, v in meta.items() if v is not None}
            
            vectors_to_upsert.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": clean_meta
            })
            
        # Upsert in batches of 100 to respect Pinecone limits
        batch_size = 100
        for i in tqdm(range(0, len(vectors_to_upsert), batch_size)):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch, namespace=col_name)
            
    print("\n✅ Migration successfully completed! Your vectors are now in the cloud.")

if __name__ == "__main__":
    migrate()
