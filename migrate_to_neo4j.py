import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
import chromadb
from tqdm import tqdm
from neo4j import GraphDatabase

load_dotenv()

EMBED_MODEL = "multilingual-e5-large"
BATCH_SIZE = 96

def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USERNAME')
    pwd = os.getenv('NEO4J_PASSWORD')
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    
    with driver.session() as session:
        # Create vector index
        print("Creating vector index on DocumentChunk(embedding)...")
        session.run("CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:DocumentChunk) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}")
        
        # Create an index on chunk_id for faster merges
        session.run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:DocumentChunk) ON (c.id)")
    
    db_path = str(Path("data/vector_store").absolute())
    client = chromadb.PersistentClient(path=db_path)
    collections = ['idx_fixed', 'idx_recursive', 'idx_semantic', 'idx_parent_child']

    for col_name in collections:
        print(f"\n=== Processing: {col_name} ===")
        try:
            col = client.get_collection(name=col_name)
        except Exception:
            print(f"  Skipping {col_name} - not found.")
            continue

        data = col.get(include=['metadatas', 'documents'])
        ids = data['ids']
        documents = data['documents']
        metadatas = data['metadatas']

        if not ids: continue
        
        print(f"  Found {len(ids)} documents. Embedding + upserting to Neo4j...")
        
        for i in tqdm(range(0, len(ids), BATCH_SIZE)):
            batch_ids = ids[i:i+BATCH_SIZE]
            batch_docs = documents[i:i+BATCH_SIZE]
            batch_metas = metadatas[i:i+BATCH_SIZE]

            try:
                result = pc.inference.embed(
                    model=EMBED_MODEL,
                    inputs=batch_docs,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                embeddings = [r.values for r in result]
            except Exception as e:
                print(f"  Embedding error at batch {i}: {e}")
                continue

            with driver.session() as session:
                for vid, emb, meta, doc in zip(batch_ids, embeddings, batch_metas, batch_docs):
                    clean_meta = {k: v for k, v in (meta or {}).items() if v is not None}
                    source = str(clean_meta.get('source', ''))
                    
                    session.run(
                        """
                        MERGE (c:DocumentChunk {id: $id})
                        SET c.text = $text,
                            c.chunk_type = $chunk_type,
                            c.metadata = $metadata,
                            c.source = $source,
                            c.embedding = $embedding
                        """,
                        id=vid, text=doc, chunk_type=col_name, metadata=str(clean_meta), source=source, embedding=emb
                    )

    print("\n=== DONE! ===")
    
    with driver.session() as session:
        count = session.run("MATCH (c:DocumentChunk) RETURN count(c) as count").single()["count"]
        print(f"Total DocumentChunk nodes in Neo4j: {count}")
        
    driver.close()

if __name__ == "__main__":
    main()
