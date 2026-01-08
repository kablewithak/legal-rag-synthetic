# Legal RAG: Hybrid Search Pipeline 🇿🇦

A local-first Retrieval-Augmented Generation (RAG) system built to solve the "Discovery Problem" in legal contracts.

## The Problem
Standard keyword search (`Ctrl+F`) fails when legal concepts use different phrasing (e.g., searching for "Indemnification" misses "Hold Harmless"). Vector search misses specific case numbers. 

## The Solution: Hybrid Search
This pipeline implements **Reciprocal Rank Fusion (RRF)** to combine:
1.  **Semantic Search:** Uses `all-MiniLM-L6-v2` vectors to find concepts.
2.  **Keyword Search:** Uses BM25 to find exact text/entities.
3.  **Refined Ingestion:** Uses Recursive Character Splitting to prevent "chopped" data (e.g., keeping "OBLIGATIONS" intact).

## 🏗 Tech Stack
* **Language:** Python 3.11
* **Database:** LanceDB (Serverless Vector DB)
* **Embeddings:** SentenceTransformers
* **Orchestration:** Manual Python (No frameworks)

## Project Structure
* `01_ingest.py`: Chunking and data preparation.
* `02_generate_qa.py`: Synthetic data generation for testing.
* `03_baseline.py`: Vector search implementation.
* `04_bm25.py`: Keyword search control group.
* `05_hybrid.py`: **Final Product** (RRF Algorithm).

##  How to Run
1.  Install dependencies: `pip install lancedb pandas sentence-transformers langchain`
2.  Run the pipeline:
    ```bash
    python 05_hybrid.py
    ```

---
*Built as part of the #MakingItIntoML journey.*