import os
import sys
import time
import logging 
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import vector_search_rrf, graph_search
from src.query_router import route_query

# Load environment variables
load_dotenv()

# Configure page layout
st.set_page_config(
    page_title="Corporate Intelligence Hub",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "companies" not in st.session_state:
    st.session_state.companies = ["Apple", "Microsoft", "NVIDIA", "Amazon", "Tesla"]

# Define Groq LLM client for Cloud compatibility
def get_llm_client():
    # Using Groq Cloud API for Hugging Face Spaces compatibility
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )

def generate_response_stream(query, hybrid_context):
    client = get_llm_client()
    system_prompt = (
        "You are a senior hedge fund market researcher. Answer the following question "
        "\n\n"
        f"Context:\n{hybrid_context}\n\n"
        f"Question:\n{query}"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192", # Cloud model
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.0,
            max_tokens=1024,
            stream=True
        )
        return response
    except Exception as e:
        return f"Error connecting to LLM API: {e}"

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("⚙️ Operational States")
    
    st.subheader("Corporate Scope")
    companies = ["Apple", "Microsoft", "NVIDIA", "Amazon", "Tesla"]
    for company in companies:
        st.checkbox(company, value=True, key=f"chk_{company}")
        
    st.markdown("---")
    st.subheader("Database Metadata")
    st.write("**Vector Store:** ChromaDB (Embedded SQLite)")
    st.write("**Graph Store:** Neo4j Aura (Cloud)")
    st.write("**Embedding Model:** all-MiniLM-L6-v2")
    
    st.markdown("---")
    st.subheader("☁️ Cloud Hardware Profile")
    st.progress(40, text="Streamlit Server RAM Utilization")

# --- MAIN DASHBOARD UI ---
st.title("Corporate Intelligence Hub")
st.markdown("### Hybrid Vector + GraphRAG Engine")

# Input Console
query_input = st.text_input(
    "Enter cross-company financial or supply chain query...",
    key="query_input"
)

col_action, _ = st.columns([1, 4])
with col_action:
    execute = st.button(" Execute Hybrid Search", use_container_width=True)

if execute and query_input:
    # 1. Start execution with status indicator
    with st.status("Executing Multi-Path Intelligence Retrieval...", expanded=True) as status:
        
        # Step A: NER Extraction
        st.write("Extracting Named Entities...")
        start_time = time.time()
        entities = route_query(query_input)
        
        # Step B: Vector Path (RRF)
        st.write("Fusing Parallel Context Arrays via RRF...")
        vector_results = vector_search_rrf(query_input, top_k=5)
        
        # Step C: Graph Path
        st.write("Traversing Graph Matrix...")
        graph_results = graph_search(entities)
        
        status.update(label=f"Retrieval Complete in {time.time() - start_time:.2f}s", state="complete", expanded=False)

    st.markdown("---")
    
    # 2. Visual Dual-Path Retrieval Dashboard
    st.markdown("### 🧩 Dual-Path Retrieval Architecture")
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Vector Store (RRF Outputs)")
        if vector_results:
            for i, text in enumerate(vector_results):
                with st.container(border=True):
                    # Badge indicators
                    st.markdown(f"**Rank {i+1}** 🏆 | `all-MiniLM-L6-v2` | `Hybrid Ensemble`")
                    st.write(text[:500] + "..." if len(text) > 500 else text)
        else:
            st.info("No semantic vector matches found.")
            
    with colB:
        st.subheader("Neo4j Factual Connections")
        if graph_results:
            with st.container(border=True):
                st.markdown("`2-Hop Cypher Traversal Results`")
                for fact in graph_results:
                    st.markdown(f"- {fact}")
        else:
            st.warning("No explicit structural relationships found in Neo4j for the extracted entities.")

    # 3. High-Performance Context Fusion
    st.markdown("---")
    fusion_parts = []
    if graph_results:
        fusion_parts.append("### Structured Knowledge Graph Facts ###")
        fusion_parts.append("\n".join(graph_results))
        fusion_parts.append("\n")
    if vector_results:
        fusion_parts.append("### Relevant Document Excerpts ###")
        fusion_parts.append("\n\n---\n\n".join(vector_results))
        
    hybrid_context = "\n".join(fusion_parts) if fusion_parts else "No relevant context found."
    
    with st.expander("🔎 View Combined Fusion Payload"):
        st.text(hybrid_context)

    # 4. Final Generation
    st.markdown("### 🧠 Final Grounded Analysis (Groq Cloud)")
    
    if hybrid_context == "No relevant context found.":
        st.error("Insufficient context to generate a factual answer.")
    else:
        # Create an empty container for typewriter effect
        response_container = st.empty()
        full_response = ""
        
        try:
            stream = generate_response_stream(query_input, hybrid_context)
            if isinstance(stream, str): # Error returned
                st.error(stream)
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        response_container.markdown(full_response + "▌")
                # Final render without cursor
                response_container.markdown(full_response)
        except Exception as e:
            st.error(f"Generation failed: {e}")
