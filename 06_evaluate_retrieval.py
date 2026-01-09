import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer
from hybrid_search import hybrid_search 

# --- SETUP ---
print("⚙️ Setting up the Referee...")
db = lancedb.connect("./lancedb_data")
table = db.open_table("nda_vector_store")
model = SentenceTransformer('all-MiniLM-L6-v2')
df_eval = pd.read_csv("eval_dataset.csv")

# --- DEFINE CONTESTANTS ---
def get_baseline_results(query, k=10):
    query_vec = model.encode(query).tolist()
    # Return Top 10 to see where it falls
    results = table.search(query_vec).limit(k).to_pandas()
    return results['chunk_id'].tolist()

def get_hybrid_results(query, k=10):
    # Hybrid usually returns fewer, but let's fetch what we can
    return hybrid_search(query)

# --- THE BATTLE ENGINE (MRR VERSION) ---
def evaluate_system(system_name, search_func):
    print(f"\n🥊 Testing {system_name}...")
    score = 0
    mrr_score = 0
    total = len(df_eval)
    
    for index, row in df_eval.iterrows():
        query = row['question']
        target_id = row['chunk_id']
        
        retrieved_ids = search_func(query)
        
        # Check Rank
        if target_id in retrieved_ids:
            rank = retrieved_ids.index(target_id) + 1 # 1-based index
            score += 1
            mrr_score += (1 / rank)
            # Only print if it's NOT rank 1 (to reduce noise)
            if rank > 1:
                print(f"  ⚠️ Q{index+1} Hit at Rank #{rank}")
        else:
            print(f"  ❌ Q{index+1} Miss (Not in results)")
            
    hit_rate = score / total
    mrr = mrr_score / total
    
    print(f"🏁 {system_name} | Hit Rate: {hit_rate:.2%} | MRR: {mrr:.3f}")
    return hit_rate, mrr

# --- RUN ---
if __name__ == "__main__":
    b_hit, b_mrr = evaluate_system("Baseline (Vector)", get_baseline_results)
    h_hit, h_mrr = evaluate_system("Hybrid (RRF)", get_hybrid_results)

    print("\n------------------------------------------------")
    print("🏆 FINAL VERDICT")
    print("------------------------------------------------")
    print(f"Vector MRR: {b_mrr:.3f}")
    print(f"Hybrid MRR: {h_mrr:.3f}")
    
    if h_mrr > b_mrr:
        print(f"✅ Hybrid Wins on Ranking Quality!")
    elif h_mrr < b_mrr:
        print(f"⚠️ Vector Wins (RRF might be diluting good results)")
    else:
        print(f"🤝 Tie.")