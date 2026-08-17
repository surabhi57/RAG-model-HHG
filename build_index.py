import pandas as pd
import chromadb
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from chunking import chunk_fixed

path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="validation/hinval.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(path)

SAMPLE_SIZE = 100
df_sample = df.iloc[:SAMPLE_SIZE]

print(f"Working with {len(df_sample)} queries")

records = []
for _, row in df_sample.iterrows():
    query_id = row["query_id"]
    query = row["query"]
    passages_dict = row["passages"]
    hindi_passages = passages_dict["Translated_passages"]
    is_selected = passages_dict["is_selected"]
    for idx, (passage_text, selected_flag) in enumerate(zip(hindi_passages, is_selected)):
        records.append({
            "text": passage_text,
            "query_id": query_id,
            "query": query,
            "passage_idx": idx,
            "is_selected": int(selected_flag)
        })

print(f"Extracted {len(records)} total passages (before chunking)")

chunked_records = []
for rec in records:
    chunks = chunk_fixed(rec["text"], chunk_size=80, overlap=20)
    for c_idx, chunk_text in enumerate(chunks):
        chunked_records.append({
            "chunk_text": chunk_text,
            "query_id": int(rec["query_id"]),
            "query": rec["query"],
            "passage_idx": int(rec["passage_idx"]),
            "is_selected": int(rec["is_selected"]),
            "chunk_idx": int(c_idx)
        })

print(f"Total chunks after chunking: {len(chunked_records)}")

print("\nLoading embedding model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Embedding chunks...")
texts = [r["chunk_text"] for r in chunked_records]
embeddings = model.encode(texts, show_progress_bar=True).tolist()

print("Storing in ChromaDB...")
client = chromadb.Client()
try:
    client.delete_collection("msmarco_chunks")
except Exception:
    pass
collection = client.create_collection("msmarco_chunks")

ids = [f"chunk_{i}" for i in range(len(chunked_records))]
metadatas = [
    {
        "query_id": r["query_id"],
        "query": r["query"],
        "passage_idx": r["passage_idx"],
        "is_selected": r["is_selected"],
        "chunk_idx": r["chunk_idx"]
    }
    for r in chunked_records
]

collection.add(
    embeddings=embeddings,
    documents=texts,
    ids=ids,
    metadatas=metadatas
)

print(f"Indexed {len(chunked_records)} chunks into ChromaDB.")

test_row = df_sample.iloc[0]
test_query = test_row["query"]
print(f"\nTest query: {test_query}")

query_embedding = model.encode([test_query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=3)

print("\nTop 3 retrieved chunks:")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n--- Result {i+1} (is_selected={meta['is_selected']}) ---")
    print(doc[:200])
