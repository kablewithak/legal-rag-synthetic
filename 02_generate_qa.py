import pandas as pd
import ollama
import json

# Load the chunks we created in step 1
df = pd.read_csv("nda_chunks.csv")

eval_dataset = []

print(f"🚀 Starting generation with Llama 3 locally...")
print(f"Processing {len(df)} chunks. Please wait...")

# --- LOOP THROUGH CHUNKS ---
for index, row in df.iterrows():
    chunk_text = row['text']
    chunk_id = row['chunk_id']
    
    # The Prompt to Llama 3
    prompt = f"""
    You are a lawyer. Read the text below and write ONE difficult question 
    that can be answered using ONLY this text.
    Also provide the exact answer found in the text.
    
    TEXT: "{chunk_text}"
    
    Return the response in this exact JSON format:
    {{
        "question": "The question here",
        "answer": "The answer here"
    }}
    """
    
    try:
        # --- CALLING OLLAMA (Lines 34-38) ---
        response = ollama.chat(model='llama3.2:3b', messages=[
            {'role': 'user', 'content': prompt},
        ], format='json') 
        
        # Parse the JSON response
        content = json.loads(response['message']['content'])
        
        eval_dataset.append({
            "chunk_id": chunk_id,
            "question": content['question'],
            "ground_truth_answer": content['answer'],
            "source_text": chunk_text
        })
        
        print(f"✅ Generated QA for chunk {index+1}/{len(df)}")
        
    except Exception as e:
        print(f"❌ Error on chunk {index}: {e}")

# --- SAVING RESULTS (Lines 53-55) ---
eval_df = pd.DataFrame(eval_dataset)
eval_df.to_csv("eval_dataset.csv", index=False)
print("\n🎉 Evaluation dataset saved to 'eval_dataset.csv'")