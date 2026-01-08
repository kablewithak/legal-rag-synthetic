import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- 1. SETUP ---
db = lancedb.connect("./lancedb_data")
table = db.open_table("nda_vector_store")

# Load the model (We need this to create vectors manually)
print("⏳ Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure FTS index exists
try:
    table.create_fts_index("text", replace=True)
except:
    pass

# --- 2. THE MANUAL HYBRID ENGINE (RRF) ---
def hybrid_search(query_text):
    """
    Manually runs Vector Search + Keyword Search and combines them
    using Reciprocal Rank Fusion (RRF).
    """
    # A. Run Vector Search (Semantic)
    query_vector = model.encode(query_text).tolist()
    # We get more candidates (limit=10) to allow for re-ranking
    vec_df = table.search(query_vector).limit(10).to_pandas()
    
    # B. Run Keyword Search (Exact)
    kw_df = table.search(query_text, query_type="fts").limit(10).to_pandas()
    
    # C. Perform Reciprocal Rank Fusion (RRF)
    # Formula: Score = 1 / (k + rank)
    scores = {}
    RRF_K = 60  # A standard constant used in search engines
    
    # 1. Score the Vector Results
    for rank, row in vec_df.iterrows():
        doc_id = row['chunk_id']
        # Rank is 0-indexed, so we add 1
        score = 1 / (RRF_K + rank + 1)
        scores[doc_id] = scores.get(doc_id, 0) + score
        
    # 2. Score the Keyword Results
    for rank, row in kw_df.iterrows():
        doc_id = row['chunk_id']
        score = 1 / (RRF_K + rank + 1)
        scores[doc_id] = scores.get(doc_id, 0) + score
    
    # D. Sort and Retrieve Top 3
    # Sort the dictionary by score (descending)
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    # Return just the Top 3 IDs
    top_ids = [doc_id for doc_id, score in sorted_docs[:3]]
    return top_ids

# --- 3. THE EXPERIMENT ---
df_eval = pd.read_csv("eval_dataset.csv")
score = 0
total_questions = len(df_eval)

print(f"\n🚀 Running Hybrid Search Experiment (Manual RRF)...")

for index, row in df_eval.iterrows():
    query = row['question']
    expected_chunk_id = row['chunk_id']
    
    # Run our custom Hybrid function
    retrieved_ids = hybrid_search(query)
    
    # CHECK RESULTS
    if expected_chunk_id in retrieved_ids:
        score += 1
        print(f"Question {index+1}: ✅ PASS")
    else:
        print(f"Question {index+1}: ❌ FAIL")
        print(f"   Query: {query}")
        print(f"   Expected: {expected_chunk_id}")
        print(f"   Got: {retrieved_ids}")

final_score = score / total_questions
print(f"\n🏆 Final Hybrid Score: {final_score:.2f} ({score}/{total_questions})")