import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """
You are a Named Entity Recognition (NER) extractor. Your task is to extract entities from the user's natural language query.

Identify explicit corporate names, industrial sectors/components, and geographical zones.
Output ONLY a raw JSON object with no markdown wrappers (e.g., no ```json) and no conversational text.

Format exactly as follows:
{
  "companies": ["Company A", "Company B"],
  "components": ["Component X", "Sector Y"],
  "locations": ["Location 1", "Location 2"]
}
If a category has no items, return an empty array for that key.
"""

def fallback_ner(query: str) -> dict:
    """
    Fallback text-matching mechanism to catch entities if the API drops or fails.
    Uses basic keyword matching against a predefined list of known entities.
    """
    logger.info("Using fallback text-matching mechanism for NER.")
    
    # A simplified mock list of entities for the fallback mechanism
    # In a production system, this would be dynamically loaded from Neo4j or a taxonomy file.
    known_companies = ["apple", "nvidia", "intel", "amd", "tsmc", "samsung", "microsoft", "google"]
    known_components = ["semiconductors", "chips", "gpus", "cpus", "silicon", "hardware", "software", "manufacturing"]
    known_locations = ["east asia", "taiwan", "china", "usa", "europe", "north america", "south korea", "asia"]
    
    query_lower = query.lower()
    
    companies = [c.title() if c != "tsmc" else "TSMC" for c in known_companies if c in query_lower]
    components = [c for c in known_components if c in query_lower]
    locations = [c.title() for c in known_locations if c in query_lower]
    
    return {
        "companies": companies,
        "components": components,
        "locations": locations
    }

def route_query(query: str) -> dict:
    """
    Parses an incoming user natural language query to extract entities using local LLM.
    Falls back to simple keyword matching on failure.
    """
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": query}
            ],
            model="gpt-oss-120b",
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        
        # Strip markdown wrappers if the model improperly includes them
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
            
        if content.endswith("```"):
            content = content[:-3]
            
        content = content.strip()
        
        parsed = json.loads(content)
        
        # Ensure all expected keys exist
        return {
            "companies": parsed.get("companies", []),
            "components": parsed.get("components", []),
            "locations": parsed.get("locations", [])
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error from local LLM: {e}. Content: {content}")
        return fallback_ner(query)
    except Exception as e:
        logger.error(f"Local LLM API or parsing failed: {e}. Falling back to text-matching.")
        return fallback_ner(query)

if __name__ == "__main__":
    # Test the router with a sample question
    test_query = "If a severe semiconductor manufacturing bottleneck occurs in East Asia, which of these five companies are most vulnerable, and what are their shared risks?"
    logger.info(f"Query: {test_query}")
    
    entities = route_query(test_query)
    logger.info(f"Extracted Entities:\n{json.dumps(entities, indent=2)}")
