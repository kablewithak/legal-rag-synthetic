import pandas as pd
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. THE RAW DATA ---
RAW_NDA_TEXT = """
NON-DISCLOSURE AGREEMENT (NDA)

1. PARTIES
This Agreement is entered into by and between TechCorp Inc. ("Disclosing Party") 
and Future Innovations LLC ("Receiving Party").

2. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" shall include all technical data, trade secrets, 
source code, financial data, and customer lists provided by the Disclosing Party. 
Information publicly known or independently developed by the Receiving Party 
is excluded from this definition.

3. OBLIGATIONS
The Receiving Party agrees to hold Confidential Information in strict confidence. 
They shall not disclose such information to any third party without prior written consent. 
The Receiving Party shall use the information solely for the purpose of the Business Relationship.

4. TERM AND TERMINATION
This Agreement shall remain in effect for a period of 2 years from the Effective Date. 
The obligations of confidentiality shall survive the termination of this Agreement 
for a period of 5 years.

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.
"""

# --- 2. THE CHUNKING LOGIC (UPGRADED) ---
def chunk_text(text, chunk_size=200, overlap=50):
    # We use the "Smart Splitter" from LangChain
    # It tries to split on paragraphs first (\n\n), then sentences (.), then words ( )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""] # Priority list
    )
    
    # This returns a list of strings
    raw_chunks = text_splitter.split_text(text)
    
    # Format them into our JSON structure
    formatted_chunks = []
    for chunk in raw_chunks:
        formatted_chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": chunk,
            "source": "NDA_Doc_v1"
        })
        
    return formatted_chunks

# --- 3. EXECUTION ---
if __name__ == "__main__":
    data_chunks = chunk_text(RAW_NDA_TEXT, chunk_size=200, overlap=50)
    df = pd.DataFrame(data_chunks)
    df.to_csv("nda_chunks.csv", index=False)
    print(f"✅ Successfully created {len(df)} chunks (Smart Split).")