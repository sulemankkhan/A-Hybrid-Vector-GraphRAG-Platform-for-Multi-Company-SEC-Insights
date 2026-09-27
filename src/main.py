import os
import sys
# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Import the core retrieval engine
from src.retriever import generate_hybrid_context

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mute noisy third-party HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def main():
    # Load all core environmental configurations securely
    load_dotenv()
    
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    # Setup Argument Parser for flexible testing via terminal
    parser = argparse.ArgumentParser(description="Hybrid RAG Search Interface")
    parser.add_argument("--query", type=str, required=True, help="The user question to answer.")
    args = parser.parse_args()
    
    user_query = args.query
    
    print("\n" + "="*60)
    print(f"Question: {user_query}")
    print("="*60 + "\n")
    
    # Execute Dual-Path Retrieval
    # `generate_hybrid_context` pulls from the Neo4j graph and all 4 ChromaDB vector stores (using RRF).
    hybrid_context = generate_hybrid_context(user_query)
    
    # Construct the highly performant generation prompt
    system_prompt = (
        "You are a senior hedge fund market researcher. Answer the following question "
        "using ONLY the verified facts provided in the Context block. If the context "
        "does not contain enough data, state clearly that information is unavailable.\n\n"
        f"Context:\n{hybrid_context}\n\n"
        f"Question:\n{user_query}"
    )
    
    # Send the compiled payload to Local LLM
    logger.info("Sending RRF fused prompt to Local LLM (llama3.2)...")
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": system_prompt}
            ],
            model="llama3.2",
            temperature=0.0,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate answer from Local LLM: {e}")
        return

    # Print out the complete structured answer and metadata to the terminal
    print("\n" + "="*60)
    print("FINAL GENERATED ANSWER")
    print("="*60)
    print(answer)
    print("\n" + "-"*60)
    print("METADATA:")
    print("- Retrieval Architecture: Reciprocal Rank Fusion (RRF) & Graph")
    print(f"- Generation Model      : llama3.2")
    print(f"- Hybrid Context Length : {len(hybrid_context)} characters")
    print("-" * 60 + "\n")

if __name__ == "__main__":
    main()

