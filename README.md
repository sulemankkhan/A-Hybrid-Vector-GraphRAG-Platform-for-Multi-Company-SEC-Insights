# Architecture & Flow

This repository implements a high-performance Hybrid Retrieval-Augmented Generation (RAG) system engineered specifically to parse, vectorize, and semantically query complex SEC 10-K financial filings. By running a dual-path architecture—combining a semantic Vector RAG layer with a structural GraphRAG layer—the system executes multi-hop financial analytics natively on constrained Apple Silicon hardware.

## 🎨 System Topology

```mermaid
graph TD
    subgraph Phase 1: Ingestion Architecture
        A[10-K HTML Files] --> B(BeautifulSoup State-Machine)
        B --> C[Text Node JSON]
        
        C --> D1[Fixed-Token]
        C --> D2[Recursive Character]
        C --> D3[Semantic]
        C --> D4[Parent-Child]
        
        D1 & D2 & D3 & D4 --> E[HuggingFace Embeddings]
        E --> F[(ChromaDB: Persistent SQLite Local Client)]
        
        C --> G[Local LLM Extraction]
        G --> H[JSON Triplets]
        H --> I[(Neo4j: Parameterized Cypher Transaction Engine)]
    end
    
    subgraph Phase 2: Retrieval & Orchestration Core
        Q[User Query] --> NER(NER Router)
        Q --> RRF(Reciprocal Rank Fusion Engine)
        
        RRF -- Parallel Vector Search --> S[Cosine Similarity Search across 4 Collections]
        S --> F
        F --> T[Semantic Text Context (Top 3 RRF Ranked)]
        
        NER -- Graph Path B --> U[Multi-Hop Graph Relational Traversal]
        U --> I
        I --> V[Declarative Knowledge Facts]
        
        T & V --> W[Hybrid Context Fusion Layer]
        W --> UI(Streamlit Interactive Dashboard)
        UI --> X[Local Llama 3.2 Generation]
        X --> Y[Final Grounded Analysis]
    end
```

## 📁 Repository Blueprint

```text
RAg/
├── data/
│   ├── raw/                 # Raw SEC 10-K HTML files
│   └── processed/           # Incremental JSON flash-saves
├── src/
│   ├── parser.py            # HTML ingestion and markup anchor tracking
│   ├── chunking.py          # Fixed and Recursive chunking algorithms
│   ├── semantic_chunking.py # Semantic and Parent-Child chunking models
│   ├── vector_ingest.py     # ChromaDB embedding population
│   ├── graph_extractor.py   # Triplet extraction via LLM
│   ├── graph_ingest.py      # Neo4j graph construction
│   ├── query_router.py      # Named Entity Recognition (NER) extraction
│   ├── retriever.py         # Reciprocal Rank Fusion (RRF) and Graph Traversal
│   ├── main.py              # CLI Execution orchestrator
│   ├── app.py               # Streamlit UI Dashboard Interface
│   └── evaluate.py          # Ragas benchmarking script
├── README.md
├── requirements.txt
└── .env
```

## Phase 1: Ingestion Architecture

### The HTML Parsing Paradigm
Unlike standard systems that ingest plain text or PDFs, this architecture relies strictly on HTML 10-K files processed via a `BeautifulSoup State-Machine` within [`src/parser.py`](src/parser.py). This constraint is mandatory for two reasons:
1. HTML structural tags preserve tabular boundaries (`<td>` and `<tr>`), keeping dense Net Sales operational matrices intact for downstream quantitative querying.
2. Heading elements (`<h1>`-`<h4>`) function as layout state anchors, allowing the script to attach active section properties directly to adjacent unstructured paragraph blocks using a `Dynamic Context Pointer Tracker`.

### The Multi-Strategy Chunking Boundary Matrix
Following ingestion, raw text nodes undergo four distinct computational slicing strategies defined in [`src/chunking.py`](src/chunking.py) and [`src/semantic_chunking.py`](src/semantic_chunking.py):

1. **Fixed-Token**: Slices blocks precisely via `cl100k_base Tiktoken Buckets` to maintain tabular matrix layouts without truncation.
2. **Recursive Character**: Executes smart paragraph/sentence backtracking down a fallback `Hierarchical Waterfall Backtracking` chain (`\n\n` → `\n` → ` `).
3. **Semantic**: Executes dynamic vector splitting based strictly on `Cosine Similarity Threshold Boundaries` to group contextually identical sentences.
4. **Parent-Child**: Implements a `Parent-Child Vector Linkage Matrix` that fetches high-precision child nodes but expands up to parent text sizes during retrieval to prevent LLM context truncation.

Processed chunks are serialized as text arrays (distinct from mathematical embeddings) before being passed to [`src/vector_ingest.py`](src/vector_ingest.py), which maps the arrays into high-dimensional vectors stored inside a `Persistent SQLite Local Client (chroma.sqlite3)`.

### Graph Knowledge Extraction & Thermal Constraints
Parallel to vector generation, [`src/graph_extractor.py`](src/graph_extractor.py) queries the local model to extract entity-relationship triplets for graph construction in [`src/graph_ingest.py`](src/graph_ingest.py). 

**Hardware Execution Profile:**
This system is hard-coded to operate within the thermal and memory constraints of a MacBook Air M1 8GB environment:
- Computations leverage the `Apple Silicon MPS (Metal Performance Shaders) Backend` to bypass standard Nvidia CUDA dependencies.
- Sub-processing is strictly limited to 3 chunks per mini-batch configuration to fit restricted local memory context windows.
- State is preserved via incremental JSON flash-saves, enabling safe system resume states (e.g., recovering immediately from chunk 60 upon failure).
- The pipeline executes a strict 10-minute (600 seconds) cooling break after 20 extraction batches to mitigate critical thermal tracking anomalies.

## Phase 2: Retrieval & Orchestration Core

Query execution is orchestrated either via the CLI ([`src/main.py`](src/main.py)) or the interactive Web Dashboard ([`src/app.py`](src/app.py)). Upon receiving a query, the architecture splits into two independent parallel retrieval operations managed by [`src/retriever.py`](src/retriever.py):

- **Vector Path (Reciprocal Rank Fusion)**: Bypasses brittle intent routing by concurrently executing cosine similarity searches against all 4 independent ChromaDB chunking indices. A mathematical `Reciprocal Rank Fusion (RRF)` algorithm re-scores and aggregates the outputs to distill the absolute top 3 winning semantic text chunks.
- **Graph Path (Multi-Hop Traversal)**: [`src/query_router.py`](src/query_router.py) executes a Named Entity Recognition (NER) pass. Extracted entities are passed to a `Parameterized Cypher Transaction Engine` to execute `Multi-Hop Graph Relational Traversal` across the Neo4j database, returning structured relational facts.

Both pipelines culminate in a combined Hybrid Context Fusion Layer. This unified payload is rendered beautifully in the `Streamlit Interactive Dashboard` and passed to the local `llama3.2` LLM to stream the final hallucination-free response, completely bypassing external cloud API dependencies. System accuracy and architectural integrity are ultimately benchmarked via Ragas metrics within [`src/evaluate.py`](src/evaluate.py).
