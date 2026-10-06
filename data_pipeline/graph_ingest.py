import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, CypherSyntaxError

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Strict whitelists to prevent Cypher injection with dynamic string formatting
ALLOWED_LABELS = {"Company", "Component", "Material", "Location", "Product", "Competitor", "Supplier"}
ALLOWED_RELS = {"DEPENDS_ON", "MANUFACTURES_IN", "COMPETES_WITH", "SUPPLIES", "EXPOSED_TO"}

def create_constraints(driver):
    """Set unique constraints on the 'name' field for all allowed node labels."""
    logger.info("Setting up database constraints/indices...")
    with driver.session() as session:
        for label in ALLOWED_LABELS:
            try:
                # Neo4j 5.x syntax
                query = f"CREATE CONSTRAINT {label}_name_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                session.run(query)
            except CypherSyntaxError:
                # Fallback for Neo4j 4.x versions
                try:
                    fallback_query = f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.name IS UNIQUE"
                    session.run(fallback_query)
                except Exception as ex:
                    logger.warning(f"Constraint creation failed for {label}: {ex}")
            except Exception as e:
                logger.error(f"Error creating constraint for {label}: {e}")

def batch_ingest(tx, triplets_batch):
    """
    Optimized batch transaction function.
    Groups triplets by signature to safely inject dynamic labels into parameterized UNWIND queries.
    """
    queries = {}
    
    for triplet in triplets_batch:
        # Protect against the LLM occasionally hallucinating nested lists instead of dicts
        if not isinstance(triplet, dict):
            logger.warning("Skipping malformed triplet (not a dict)")
            continue
            
        s_type = triplet.get("type")
        t_type = triplet.get("target_type")
        rel = triplet.get("relationship")
        
        s_name = triplet.get("source")
        t_name = triplet.get("target")
        
        if not all([s_type, t_type, rel, s_name, t_name]):
            continue
            
        if s_type not in ALLOWED_LABELS or t_type not in ALLOWED_LABELS or rel not in ALLOWED_RELS:
            logger.warning(f"Skipping invalid label/relationship in triplet: {triplet}")
            continue
            
        # Group by signature to reuse the same cypher string
        sig = (s_type, t_type, rel)
        if sig not in queries:
            queries[sig] = []
            
        queries[sig].append({"source_name": s_name, "target_name": t_name})
        
    # Execute batch UNWIND queries for each signature
    for (s_type, t_type, rel), params_list in queries.items():
        query = f"""
        UNWIND $batch AS param
        MERGE (s:{s_type} {{name: param.source_name}})
        MERGE (t:{t_type} {{name: param.target_name}})
        MERGE (s)-[r:{rel}]->(t)
        """
        try:
            tx.run(query, batch=params_list)
        except CypherSyntaxError as e:
            logger.error(f"Invalid Cypher Syntax: {e} | Query: {query}")
        except Exception as e:
            logger.error(f"Execution Error during batch ingest: {e}")

def run_graph_ingest():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    input_file = Path("data/processed/graph_triplets.json")
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
        
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON data: {e}")
        return

    logger.info("Connecting to Neo4j database...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except ServiceUnavailable as e:
        logger.error(f"Database connection timeout or unavailable: {e}")
        return
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return

    # 4. Set unique constraints
    create_constraints(driver)
    
    batch_size = 1000
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    logger.info(f"Starting ingestion of {len(data)} triplets in {total_batches} batches...")
    
    with driver.session() as session:
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            try:
                session.execute_write(batch_ingest, batch)
                logger.info(f"Processed batch {i // batch_size + 1}/{total_batches}")
            except Exception as e:
                logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")
                
    driver.close()
    logger.info("Graph ingestion completed successfully.")

if __name__ == "__main__":
    run_graph_ingest()
