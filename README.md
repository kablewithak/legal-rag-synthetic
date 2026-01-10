# Legal RAG: Hybrid Search Pipeline 🇿🇦

A local-first Retrieval-Augmented Generation (RAG) system built to solve the "Discovery Problem" in legal contracts.

## The Problem
Standard keyword search (`Ctrl+F`) fails when legal concepts use different phrasing (e.g., searching for "Indemnification" misses "Hold Harmless"). Vector search misses specific case numbers or exact codes.

## The Solution: Hybrid Search
This pipeline implements **Reciprocal Rank Fusion (RRF)** to combine:
1.  **Semantic Search:** Uses `all-MiniLM-L6-v2` vectors to find concepts.
2.  **Keyword Search:** Uses BM25 to find exact text/entities.
3.  **Adversarial Evaluation:** We stress-test the system with "Needle in a Haystack" queries and Lexical Mismatches.

## 🏗 Tech Stack
* **Language:** Python 3.11
* **Database:** LanceDB (Serverless Vector DB)
* **Embeddings:** SentenceTransformers
* **Orchestration:** Manual Python (No frameworks)

## Project Structure
### Core Pipeline
* `01_ingest.py`: Chunking and data preparation.
* `02_generate_qa.py`: Synthetic data generation (Llama 3).
* `03_baseline.py`: Vector search implementation.
* `04_bm25.py`: Keyword search control group.
* `05_hybrid.py`: **Final Product** (RRF Algorithm).

### Evaluation & Adversarial Testing
* `06_evaluate_retrieval.py`: The "Referee". Calculates Hit Rate and **MRR (Mean Reciprocal Rank)** to judge ranking quality.
* `07_add_adversarial_data.py`: Injects "Lexical Mismatches" (slang/synonyms) to confuse Keyword Search.
* `08_inject_needle.py`: Injects "Exact Codes" (e.g., `PROJECT-CHIMERA-7`) to confuse Vector Search.

## 📊 Evaluation Metrics
We use two metrics to decide the winner:
1.  **Hit Rate:** Did the answer appear in the top 10? (Recall)
2.  **MRR (Mean Reciprocal Rank):** How high was the answer? (Precision)
    * *Rank 1 = 1.0 score*
    * *Rank 2 = 0.5 score*
    * *Rank 10 = 0.1 score*

##  How to Run
1.  **Ingest Data:** `python 01_ingest.py`
2.  **Run Evaluation:** `python 06_evaluate_retrieval.py`
3.  **Inject Attacks:** `python 08_inject_needle.py`