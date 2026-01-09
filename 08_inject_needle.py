import lancedb
import pandas as pd
import uuid

# --- CONFIGURATION ---
DB_PATH = "./lancedb_data"
TABLE_NAME = "nda_vector_store"
CSV_FILE = "eval_dataset.csv"

# 1. Connect to DB
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

# 2. The Needle (Zero Semantic Context)
# "Project Chimera" is a specific entity. "Classified" is generic.
needle_text = "The secret project code is 'PROJECT-CHIMERA-7'. This file is classified."
needle_id = str(uuid.uuid4())

print(f"💉 Injecting Needle: {needle_text}")

# 3. Add to LanceDB
# Note: We must match the schema (text, chunk_id, vector, source)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
needle_vector = model.encode(needle_text).tolist()

new_chunk = [{
    "chunk_id": needle_id,
    "text": needle_text,
    "source": "Injection_Script",
    "vector": needle_vector
}]

table.add(new_chunk)

# 4. Add the Test Case to CSV
df = pd.read_csv(CSV_FILE)
new_row = {
    "chunk_id": needle_id,
    "question": "What is the secret project code?", # Semantically vague
    "ground_truth_answer": "PROJECT-CHIMERA-7",
    "source_text": needle_text
}
# Also add the EXACT CODE query to punish Vector's fuzziness
new_row_2 = {
    "chunk_id": needle_id,
    "question": "PROJECT-CHIMERA-7", # Pure Keyword
    "ground_truth_answer": "PROJECT-CHIMERA-7",
    "source_text": needle_text
}

new_df = pd.DataFrame([new_row, new_row_2])
df_updated = pd.concat([df, new_df], ignore_index=True)
df_updated.to_csv(CSV_FILE, index=False)

print(f"✅ Needle Injected. Total Test Cases: {len(df_updated)}")