import os
import json
import uuid
import logging
from pathlib import Path
import tiktoken

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def fixed_size_token_chunking(data: list[dict], max_tokens=512, overlap=50) -> list[dict]:
    chunks_out = []
    grouped_text = {}
    
    for item in data:
        key = (item.get("company", "Unknown"), item.get("section", "Unknown"))
        if key not in grouped_text:
            grouped_text[key] = []
        grouped_text[key].append(item.get("text_content", ""))
        
    for (company, section), text_list in grouped_text.items():
        combined_text = "\n".join(text_list)
        text_chunks = chunk_text_by_tokens(combined_text, max_tokens, overlap)
        
        for chunk_text in text_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text: continue
            chunks_out.append({
                "chunk_id": str(uuid.uuid4()),
                "company": company,
                "strategy": "fixed_token",
                "text": chunk_text,
                "source_section": section
            })
            
    return chunks_out

def split_text_recursively(text: str, max_chars: int, overlap: int, separators: list[str]) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    active_sep = separators[-1]
    for sep in separators:
        if sep == "":
            active_sep = sep
            break
        if sep in text:
            active_sep = sep
            break

    splits = list(text) if active_sep == "" else text.split(active_sep)
    
    chunks = []
    current_chunk_splits = []
    current_len = 0

    for split in splits:
        if len(split) > max_chars and active_sep != "":
            if current_chunk_splits:
                chunks.append(active_sep.join(current_chunk_splits))
                current_chunk_splits = []
                current_len = 0
                
            sub_seps = separators[separators.index(active_sep)+1:]
            sub_chunks = split_text_recursively(split, max_chars, overlap, sub_seps)
            chunks.extend(sub_chunks)
            continue
            
        split_len = len(split) if active_sep == "" else len(split) + len(active_sep)
        
        if current_len + split_len > max_chars and current_chunk_splits:
            chunks.append(active_sep.join(current_chunk_splits))
            
            overlap_splits = []
            overlap_len = 0
            for s in reversed(current_chunk_splits):
                s_len = len(s) if active_sep == "" else len(s) + len(active_sep)
                if overlap_len + s_len > overlap:
                    break
                overlap_splits.insert(0, s)
                overlap_len += s_len
                
            current_chunk_splits = overlap_splits
            current_len = overlap_len
            
        current_chunk_splits.append(split)
        current_len += len(split) if active_sep == "" else len(split) + len(active_sep)
        
    if current_chunk_splits:
        chunks.append(active_sep.join(current_chunk_splits))
        
    return chunks

def recursive_character_chunking(data: list[dict], max_chars=1000, overlap=100) -> list[dict]:
    separators = ["\n\n", "\n", " ", ""]
    chunks_out = []
    
    grouped_text = {}
    for item in data:
        key = (item.get("company", "Unknown"), item.get("section", "Unknown"))
        if key not in grouped_text:
            grouped_text[key] = []
        grouped_text[key].append(item.get("text_content", ""))
        
    for (company, section), text_list in grouped_text.items():
        combined_text = "\n\n".join(text_list)
        text_chunks = split_text_recursively(combined_text, max_chars, overlap, separators)
        
        for chunk_text in text_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text: continue
            chunks_out.append({
                "chunk_id": str(uuid.uuid4()),
                "company": company,
                "strategy": "recursive_character",
                "text": chunk_text,
                "source_section": section
            })
            
    return chunks_out

def process_chunks(processed_dir="data/processed"):
    processed_path = Path(processed_dir)
    json_files = list(processed_path.glob("*.json"))
    
    # Exclude output files if they already exist
    json_files = [f for f in json_files if f.name not in ["fixed_chunks.json", "recursive_chunks.json"]]
    
    if not json_files:
        logger.warning(f"No parsed JSON files found in {processed_dir}")
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
    
    # Fixed-Size Token Chunking
    logger.info("Starting Fixed-Size Token Chunking...")
    fixed_chunks = fixed_size_token_chunking(all_data, max_tokens=512, overlap=50)
    logger.info(f"Generated {len(fixed_chunks)} fixed-size token chunks.")
    
    with open(processed_path / "fixed_chunks.json", "w", encoding="utf-8") as f:
        json.dump(fixed_chunks, f, indent=2, ensure_ascii=False)
        
    # Recursive Character Chunking
    logger.info("Starting Recursive Character Chunking...")
    recursive_chunks = recursive_character_chunking(all_data, max_chars=1000, overlap=100)
    logger.info(f"Generated {len(recursive_chunks)} recursive character chunks.")
    
    with open(processed_path / "recursive_chunks.json", "w", encoding="utf-8") as f:
        json.dump(recursive_chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_chunks()
