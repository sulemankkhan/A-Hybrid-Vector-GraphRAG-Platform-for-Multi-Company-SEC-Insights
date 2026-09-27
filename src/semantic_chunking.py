import os
import json
import uuid
import logging
import re
import gc
import numpy as np
from pathlib import Path
import tiktoken
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_into_sentences(text: str) -> list[str]:
    # Basic regex for sentence boundary detection
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_chunking(data: list[dict], percentile_threshold=95) -> list[dict]:
    model_name = 'all-MiniLM-L6-v2'
    logger.info(f"Loading {model_name} for semantic chunking...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    
    chunks_out = []
    
    # Group text by company and section
    grouped_text = {}
    for item in data:
        key = (item.get("company", "Unknown"), item.get("section", "Unknown"))
        if key not in grouped_text:
            grouped_text[key] = []
        grouped_text[key].append(item.get("text_content", ""))
        
    for (company, section), text_list in grouped_text.items():
        combined_text = "\n".join(text_list)
        sentences = split_into_sentences(combined_text)
        
        if len(sentences) <= 1:
            if sentences:
                chunks_out.append({
                    "chunk_id": str(uuid.uuid4()),
                    "company": company,
                    "strategy": "semantic",
                    "text": sentences[0],
                    "source_section": section
                })
            continue
            
        # Generate embeddings in batches to save memory
        embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)
        
        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
            
        # Calculate distance
        distances = 1 - np.array(similarities)
        
        # Set dynamic threshold
        threshold = np.percentile(distances, percentile_threshold)
        
        # Chunking based on threshold
        current_chunk = [sentences[0]]
        for i, dist in enumerate(distances):
            if dist >= threshold:
                # Topic shift, cut a chunk here
                chunk_text = " ".join(current_chunk)
                chunks_out.append({
                    "chunk_id": str(uuid.uuid4()),
                    "company": company,
                    "strategy": "semantic",
                    "text": chunk_text,
                    "source_section": section
                })
                current_chunk = [sentences[i+1]]
            else:
                current_chunk.append(sentences[i+1])
                
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks_out.append({
                "chunk_id": str(uuid.uuid4()),
                "company": company,
                "strategy": "semantic",
                "text": chunk_text,
                "source_section": section
            })
            
    # Clean memory management to prevent CPU/GPU crashes
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return chunks_out

def chunk_text_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
        if i + max_tokens >= len(tokens):
            break
        step = max_tokens - overlap
        i += step if step > 0 else max_tokens
    return chunks

def hierarchical_chunking(data: list[dict], parent_tokens=1024, child_tokens=128) -> list[dict]:
    chunks_out = []
    
    # Group text by company and section
    grouped_text = {}
    for item in data:
        key = (item.get("company", "Unknown"), item.get("section", "Unknown"))
        if key not in grouped_text:
            grouped_text[key] = []
        grouped_text[key].append(item.get("text_content", ""))
        
    for (company, section), text_list in grouped_text.items():
        combined_text = "\n".join(text_list)
        
        # Parent blocks of 1024 tokens
        parent_texts = chunk_text_by_tokens(combined_text, parent_tokens, overlap=0)
        
        for p_text in parent_texts:
            parent_id = str(uuid.uuid4())
            chunks_out.append({
                "chunk_id": parent_id,
                "company": company,
                "strategy": "hierarchical_parent",
                "text": p_text,
                "source_section": section,
                "is_parent": True,
                "parent_id": None
            })
            
            # Child chunks of 128 tokens
            child_texts = chunk_text_by_tokens(p_text, child_tokens, overlap=0)
            for c_text in child_texts:
                chunks_out.append({
                    "chunk_id": str(uuid.uuid4()),
                    "company": company,
                    "strategy": "hierarchical_child",
                    "text": c_text,
                    "source_section": section,
                    "is_parent": False,
                    "parent_id": parent_id
                })
                
    return chunks_out

def process_advanced_chunks(processed_dir="data/processed"):
    processed_path = Path(processed_dir)
    json_files = list(processed_path.glob("*.json"))
    
    exclude_files = ["fixed_chunks.json", "recursive_chunks.json", "semantic_chunks.json", "hierarchical_chunks.json"]
    json_files = [f for f in json_files if f.name not in exclude_files]
    
    if not json_files:
        logger.warning(f"No original parsed JSON files found in {processed_dir}")
        return
        
    all_data = []
    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            logger.error(f"Error reading {fpath.name}: {e}")
            
    logger.info(f"Loaded {len(all_data)} text nodes from {len(json_files)} files.")
    
    logger.info("Starting Hierarchical (Parent-Child) Chunking...")
    hierarchical_chunks = hierarchical_chunking(all_data, parent_tokens=1024, child_tokens=128)
    logger.info(f"Generated {len(hierarchical_chunks)} hierarchical chunks.")
    
    with open(processed_path / "hierarchical_chunks.json", "w", encoding="utf-8") as f:
        json.dump(hierarchical_chunks, f, indent=2, ensure_ascii=False)

    del hierarchical_chunks
    gc.collect()

    logger.info("Starting Semantic Chunking...")
    semantic_chunks = semantic_chunking(all_data, percentile_threshold=95)
    logger.info(f"Generated {len(semantic_chunks)} semantic chunks.")
    
    with open(processed_path / "semantic_chunks.json", "w", encoding="utf-8") as f:
        json.dump(semantic_chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_advanced_chunks()
