import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- 1. SETUP & INGESTION ---
# Load Data
df_chunks = pd.read_csv("nda_chunks.csv")
db = lancedb.connect("./lancedb_data")

# Load the AI Brain (The Embedding Model)
print("⏳ Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# MANUAL STEP: Turn text into numbers (Vectors)
# This removes the "magic" that was failing. We do it explicitly.
print("⏳ Generating vectors for chunks...")
# We create a new column 'vector' where every row is a list of numbers
df_chunks['vector'] = df_chunks['text'].apply(lambda x: model.encode(x).tolist())

# Create the Table
table_name = "nda_vector_store"
print("⏳ Saving to LanceDB...")

# We don't need a Schema class anymore because the data is already perfect.
table = db.create_table(table_name, data=df_chunks, mode="overwrite")

print(f"✅ Indexed {len(df_chunks)} chunks.")

# --- 2. THE EXPERIMENT ---
df_eval = pd.read_csv("eval_dataset.csv")
score = 0
total_questions = len(df_eval)

print(f"\n🚀 Running Baseline Experiment (Vector Search)...")

for index, row in df_eval.iterrows():
    query_text = row['question']
    expected_chunk_id = row['chunk_id']
    
    # MANUAL STEP: Turn the Question into a Vector
    query_vector = model.encode(query_text).tolist()
    
    # SEARCH: We pass the VECTOR, not the text
    results = table.search(query_vector).limit(3).to_pandas()
    
    # CHECK RESULTS
    retrieved_ids = results['chunk_id'].tolist()
    
    if expected_chunk_id in retrieved_ids:
        score += 1
        print(f"Question {index+1}: ✅ PASS")
    else:
        print(f"Question {index+1}: ❌ FAIL")
        print(f"   Expected: {expected_chunk_id}")
        print(f"   Got: {retrieved_ids}")

final_score = score / total_questions
print(f"\n🏆 Final Recall@3 Score: {final_score:.2f} ({score}/{total_questions})")