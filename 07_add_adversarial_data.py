import pandas as pd

# --- CONFIGURATION ---
CSV_FILE = "eval_dataset.csv"

# 1. Load the existing exam
print(f"📂 Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# 2. Define the "Killer Query"
# This targets Chunk ID: 470db91f... (The Termination Clause)
# We found this ID by looking at your eval_dataset.csv file content
new_row = {
    "chunk_id": "470db91f-527f-408a-9cf0-fdbfcece9c46",
    "question": "How long are we on the hook for keeping secrets after we break up?",
    "ground_truth_answer": "5 years",
    "source_text": "4. TERM AND TERMINATION\nThis Agreement shall remain in effect for a period of 2 years from the Effective Date. \nThe obligations of confidentiality shall survive the termination of this Agreement"
}

# 3. Append safely
new_df = pd.DataFrame([new_row])
df_updated = pd.concat([df, new_df], ignore_index=True)

# 4. Save
df_updated.to_csv(CSV_FILE, index=False)

print(f"✅ Added Killer Query. Total Test Cases: {len(df_updated)}")
print(f"😈 Adversarial Question: '{new_row['question']}'")