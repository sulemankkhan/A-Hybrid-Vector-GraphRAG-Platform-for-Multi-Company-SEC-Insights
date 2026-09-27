import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """
You are a strict deterministic relationship extraction parser. Your sole task is to analyze the provided text and extract knowledge graph triplets based on specific constraints.

Allowed Node Types:
[Company, Component, Material, Location, Product, Competitor, Supplier]

Allowed Relationships:
[DEPENDS_ON, MANUFACTURES_IN, COMPETES_WITH, SUPPLIES, EXPOSED_TO]

Output Constraints:
You must output ONLY a raw JSON array of objects without any markdown wrappers (e.g., no ```json), introductory text, or pleasantries. If no relationships are found, return an empty array: []

Format exactly as follows:
[
  {
    "source": "Apple",
    "type": "Company",
    "relationship": "DEPENDS_ON",
    "target": "Semiconductors",
    "target_type": "Component"
  }
]
"""

def extract_triplets(client: OpenAI, text_batch: str) -> list:
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": f"Extract graph triplets from the following text:\n\n{text_batch}"}
            ],
            model="llama3.2",
            temperature=0.0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content.strip()
        
        # Guard against models incorrectly returning markdown wrappers
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
            
        if content.endswith("```"):
            content = content[:-3]
            
        content = content.strip()
        
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parsing Error: {e} | Content received: {content[:100]}...")
        return []
    except Exception as e:
        logger.error(f"Local LLM API Error: {e}")
        return []

def run_extraction():
    processed_dir = Path("data/processed")
    input_file = processed_dir / "recursive_chunks.json"
    output_file = processed_dir / "graph_triplets.json"
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # RESUME FROM WHERE WE LEFT OFF (Chunk 60)
    data = data[60:]
        
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    # Load existing triplets so we don't overwrite the first 72 we just extracted!
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            all_triplets = json.load(f)
    else:
        all_triplets = []
        
    # Batch text segments to optimize calls
    batch_size = 3
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        text_batch = "\n\n---\n\n".join([item["text"] for item in batch])
        
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num} of {(len(data) + batch_size - 1) // batch_size}...")
        
        triplets = extract_triplets(client, text_batch)
        if triplets:
            all_triplets.extend(triplets)
            
        # Incremental save so you don't lose progress if the script stops during a break!
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_triplets, f, indent=2, ensure_ascii=False)
            
        # Cool-down logic: 10 minutes break every 20 batches
        if batch_num % 20 == 0:
            logger.info("Laptop cooling down for 10 minutes (600 seconds)...")
            time.sleep(600)
        
    logger.info(f"Successfully extracted {len(all_triplets)} triplets and saved to {output_file}")

if __name__ == "__main__":
    run_extraction()
