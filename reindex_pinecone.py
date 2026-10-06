"""
Re-index all SEC documents into a new 1024-dim Pinecone index
using Pinecone's own inference API (multilingual-e5-large).
No HuggingFace dependency — works everywhere including Vercel.
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import chromadb
from tqdm import tqdm

load_dotenv()

NEW_INDEX_NAME = "hybridrag-1024"
EMBED_MODEL = "multilingual-e5-large"
BATCH_SIZE = 96  # Pinecone inference batch limit

def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Create new 1024-dim index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if NEW_INDEX_NAME not in existing:
        print(f"Creating new index '{NEW_INDEX_NAME}' (1024-dim, cosine)...")
        pc.create_index(
            name=NEW_INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(NEW_INDEX_NAME).status.ready:
            time.sleep(2)
        print("Index ready!")
    else:
        print(f"Index '{NEW_INDEX_NAME}' already exists, continuing...")

    index = pc.Index(NEW_INDEX_NAME)

    # Connect to local ChromaDB
    db_path = str(Path("data/vector_store").absolute())
    client = chromadb.PersistentClient(path=db_path)
    collections = ['idx_fixed', 'idx_recursive', 'idx_semantic', 'idx_parent_child']

    for col_name in collections:
        print(f"\n=== Processing: {col_name} ===")
        try:
            col = client.get_collection(name=col_name)
        except Exception:
            print(f"  Skipping {col_name} - not found in ChromaDB.")
            continue

        data = col.get(include=['metadatas', 'documents'])
        ids = data['ids']
        documents = data['documents']
        metadatas = data['metadatas']

        if not ids:
            print(f"  {col_name} is empty.")
            continue

        print(f"  Found {len(ids)} documents. Embedding + upserting...")

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

            vectors = []
            for vid, emb, meta, doc in zip(batch_ids, embeddings, batch_metas, batch_docs):
                clean_meta = {k: v for k, v in (meta or {}).items() if v is not None}
                clean_meta['text'] = doc
                vectors.append({"id": vid, "values": emb, "metadata": clean_meta})

            try:
                index.upsert(vectors=vectors, namespace=col_name)
            except Exception as e:
                print(f"  Upsert error at batch {i}: {e}")

    stats = index.describe_index_stats()
    print(f"\n=== DONE! ===")
    print(f"Index '{NEW_INDEX_NAME}': {stats.total_vector_count} total vectors")
    for ns, info in stats.namespaces.items():
        print(f"  {ns}: {info.vector_count} vectors")
    print(f"\nNext: update PINECONE_INDEX_NAME to '{NEW_INDEX_NAME}' in .env and Vercel")

if __name__ == "__main__":
    main()
