import os
import sys


try:
    import langchain_google_vertexai
    sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai
except ImportError:
    pass
# -----------------------------------------

# Add the project root to the Python path to resolve 'src' module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.retriever import generate_hybrid_context

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
load_dotenv()

# 1. Hardcode 15 complex cross-document financial analytics test questions and ground-truth answers
EVAL_DATASET = [
    {
        "question": "If a severe semiconductor manufacturing bottleneck occurs in East Asia, which of these five companies are most vulnerable, and what are their shared risks?",
        "ground_truth": "Apple and NVIDIA are highly vulnerable due to heavy dependence on TSMC in Taiwan. Shared risks include revenue loss and delayed product cycles."
    },
    {
        "question": "How does an increase in global lithium prices impact Tesla and its battery suppliers?",
        "ground_truth": "Increased lithium prices compress Tesla's margins and force suppliers like Panasonic to renegotiate contracts or absorb costs."
    },
    {
        "question": "What is the primary competitive advantage of AMD over Intel in the datacenter market?",
        "ground_truth": "AMD's primary advantage is its chiplet architecture and higher core counts offering better performance-per-watt."
    },
    {
        "question": "Which cloud provider, AWS or Azure, has more exposure to specialized AI hardware shortages?",
        "ground_truth": "Azure has significant exposure due to its massive exclusive deals with OpenAI requiring heavy H100 GPU clusters."
    },
    {
        "question": "What are the structural dependencies between ASML and global foundries?",
        "ground_truth": "ASML is the sole supplier of EUV lithography machines, making all advanced global foundries like TSMC and Samsung strictly dependent on them."
    },
    {
        "question": "How do recent export restrictions to China affect NVIDIA's quarterly revenue projections?",
        "ground_truth": "Export restrictions prevent NVIDIA from selling flagship chips to China, risking up to 20% of their data center revenue."
    },
    {
        "question": "What are the shared geopolitical risks for Apple's hardware supply chain and Microsoft's Surface line?",
        "ground_truth": "Both rely heavily on Chinese manufacturing and Taiwanese silicon, exposing them to tariff wars and regional instability."
    },
    {
        "question": "Analyze the financial impact of a potential fab closure in Texas on Samsung's memory division.",
        "ground_truth": "A fab closure would severely constrain NAND/DRAM supply, hurting Samsung's market share and temporarily boosting competitor pricing power."
    },
    {
        "question": "Identify the primary materials required for solid-state batteries and the leading corporate investors.",
        "ground_truth": "Solid-state batteries rely heavily on solid electrolytes and lithium metal anodes. Toyota and QuantumScape are leading investors."
    },
    {
        "question": "How does Alphabet's reliance on custom TPUs insulate it from NVIDIA's supply constraints?",
        "ground_truth": "Alphabet's internal TPUs handle vast portions of their AI workloads, reducing their dependency on the constrained open market for NVIDIA GPUs."
    },
    {
        "question": "What role does European regulation play in the competitive dynamics between Meta and Apple?",
        "ground_truth": "The EU's Digital Markets Act forces Apple to open up its App Store, directly benefiting Meta's advertising and app distribution strategies."
    },
    {
        "question": "Compare the capital expenditure of Meta versus Amazon on AI infrastructure for 2024.",
        "ground_truth": "Both are spending tens of billions, but Meta is focused on massive GPU clusters for Llama, while Amazon is scaling custom Trainium chips alongside GPUs."
    },
    {
        "question": "What are the primary operational vulnerabilities of Intel's new foundry business model?",
        "ground_truth": "Intel's foundry model risks include massive capital requirements, delayed process nodes, and the challenge of winning trust from competitors to manufacture their chips."
    },
    {
        "question": "How do rising energy costs in Europe affect local automotive manufacturers transitioning to EVs?",
        "ground_truth": "High energy costs increase domestic manufacturing expenses for EV batteries, putting European automakers at a price disadvantage."
    },
    {
        "question": "What is the strategic relationship between ARM's licensing model and Qualcomm's mobile dominance?",
        "ground_truth": "Qualcomm dominates mobile by designing chips based on ARM's instruction set, though they are currently in litigation over custom core licensing."
    }
]

def generate_answer(query: str, context: str) -> str:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    system_prompt = (
        "You are a senior hedge fund market researcher. Answer the following question "
        "using ONLY the verified facts provided in the Context block. If the context "
        "does not contain enough data, state clearly that information is unavailable.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": system_prompt}],
            model="llama3.2",
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate answer from Local LLM: {e}")
        return "Error generating answer."

def evaluate_strategy(strategy: str, local_llm, hf_embeddings):
    logger.info(f"--- Running automated evaluation loop for strategy: {strategy} ---")
    
    questions = []
    contexts_list = []
    answers = []
    ground_truths = []
    
    # REDUCED TO 3 QUESTIONS TO PREVENT LAPTOP OVERHEATING
    for item in EVAL_DATASET[:3]:
        q = item["question"]
        gt = item["ground_truth"]
        
        # 1. Retrieve hybrid context
        hybrid_context = generate_hybrid_context(q, vector_strategy=strategy)
        
        # 2. Generate answer
        ans = generate_answer(q, hybrid_context)
        
        questions.append(q)
        # Ragas expects contexts as a list of strings per question
        contexts_list.append([hybrid_context])
        answers.append(ans)
        ground_truths.append(gt)
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    logger.info("Calculating Ragas metrics (Context Precision, Faithfulness, Answer Relevance)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
            ],
            llm=local_llm,
            embeddings=hf_embeddings
        )
        
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        return scores
    except Exception as e:
        logger.error(f"Ragas evaluation failed for {strategy}: {e}")
        return {"context_precision": 0.0, "faithfulness": 0.0, "answer_relevancy": 0.0}

def main():
    # Setup LLM and Embeddings to use Local LLM and HuggingFace
    logger.info("Initializing Local LLM (Ollama) and HuggingFace Embeddings for Ragas evaluators...")
    local_llm = ChatOpenAI(model="llama3.2", openai_api_key="ollama", openai_api_base="http://localhost:11434/v1")
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    strategies = ["fixed", "recursive", "semantic", "parent_child"]
    results_summary = []

    for strategy in strategies:
        scores = evaluate_strategy(strategy, local_llm, hf_embeddings)
        scores["Strategy"] = strategy
        results_summary.append(scores)
        
        # Cool down the laptop after each massive Ragas strategy evaluation
        import time
        logger.info("Laptop cooling down for 5 minutes (300 seconds)...")
        time.sleep(300)
        
    # 4. Compile averages and print markdown-compatible comparison table
    df = pd.DataFrame(results_summary)
    
    cols = ["Strategy", "context_precision", "faithfulness", "answer_relevancy"]
    df = df[[c for c in cols if c in df.columns]]
    
    df = df.rename(columns={
        "context_precision": "Context Precision",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance"
    })
    
    print("\n" + "="*80)
    print("🚀 RETRIEVAL STRATEGY EVALUATION RESULTS")
    print("="*80 + "\n")
    print(df.to_markdown(index=False, floatfmt=".3f"))
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
