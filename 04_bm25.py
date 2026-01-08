import lancedb
import pandas as pd

# --- 1. SETUP ---
# Connect to the SAME database (dirty data and all)
db = lancedb.connect("./lancedb_data")
table = db.open_table("nda_vector_store")

# Create the Full Text Search (FTS) Index
print("⏳ Creating Keyword Index (BM25)...")
try:
    table.create_fts_index("text", replace=True)
except Exception as e:
    print(f"⚠️ Index Warning: {e}")

# --- 2. THE EXPERIMENT (BM25) ---
df_eval = pd.read_csv("eval_dataset.csv")
score = 0
total_questions = len(df_eval)

print(f"\n🚀 Running Control Group Experiment (Keyword/BM25)...")

for index, row in df_eval.iterrows():
    query = row['question']
    expected_chunk_id = row['chunk_id']
    
    # ... inside the loop ...
    try:
        results = table.search(query, query_type="fts").limit(1).to_pandas() # Limit 1 to see top match
        
        # CHECK RESULTS
        if not results.empty:
            top_match_id = results.iloc[0]['chunk_id']
            top_match_text = results.iloc[0]['text'][:100] # First 100 chars for preview
            
            if top_match_id == expected_chunk_id:
                score += 1
                # ONLY PRINT THIS FOR QUESTION 3 (The tricky one)
                if "OBLIGAT" in query: 
                    print(f"\n🔎 INSPECTING QUESTION {index+1}:")
                    print(f"   Query: '{query}'")
                    print(f"   Found: '...{top_match_text}...'")
                    print("   Verdict: ✅ IT MATCHED THE TYPO!")
                else:
                    print(f"Question {index+1}: ✅ PASS")
            else:
                print(f"Question {index+1}: ❌ FAIL")
        else:
            print(f"Question {index+1}: ❌ FAIL (No results)")
            
    except Exception as e:
        print(f"Question {index+1}: ⚠️ ERROR")

# --- 3. FINAL REPORT ---
final_score = score / total_questions
print(f"\n🏆 Final BM25 Score: {final_score:.2f} ({score}/{total_questions})")