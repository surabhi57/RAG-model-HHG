import pandas as pd
import chromadb
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from chunking import chunk_fixed, chunk_fixed_no_overlap, chunk_sentence_aware

SAMPLE_SIZE = 150
NUM_TEST_QUERIES = 150

print("Loading embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Loading dataset...")
path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="validation/hinval.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(path)
df_sample = df.iloc[:SAMPLE_SIZE]

records = []
for _, row in df_sample.iterrows():
    passages_dict = row["passages"]
    hindi_passages = passages_dict["Translated_passages"]
    is_selected = passages_dict["is_selected"]
    for idx, (passage_text, selected_flag) in enumerate(zip(hindi_passages, is_selected)):
        records.append({
            "text": passage_text,
            "query_id": int(row["query_id"]),
            "is_selected": int(selected_flag)
        })

strategies = {
    "fixed_overlap": chunk_fixed,
    "fixed_no_overlap": chunk_fixed_no_overlap,
    "sentence_aware": chunk_sentence_aware,
}

client = chromadb.Client()
collections = {}

for name, chunk_fn in strategies.items():
    print(f"\nBuilding index for strategy: {name}")
    chunked_records = []
    for rec in records:
        chunks = chunk_fn(rec["text"])
        for c_idx, chunk_text in enumerate(chunks):
            chunked_records.append({
                "chunk_text": chunk_text,
                "query_id": rec["query_id"],
                "is_selected": rec["is_selected"],
            })

    texts = [r["chunk_text"] for r in chunked_records]
    print(f"  {len(texts)} chunks, embedding...")
    embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()

    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name)

    ids = [f"{name}_chunk_{i}" for i in range(len(chunked_records))]
    metadatas = [{"query_id": r["query_id"], "is_selected": r["is_selected"]} for r in chunked_records]

    collection.add(embeddings=embeddings, documents=texts, ids=ids, metadatas=metadatas)
    collections[name] = collection
    print(f"  Indexed {len(chunked_records)} chunks.")

test_df = df.iloc[:NUM_TEST_QUERIES]

results_summary = {name: {"top1": 0, "top3": 0, "evaluable": 0} for name in strategies}

for _, row in test_df.iterrows():
    query = row["query"]
    passages_dict = row["passages"]
    is_selected = list(passages_dict["is_selected"])

    if 1 not in is_selected:
        continue

    query_embedding = embed_model.encode([query]).tolist()

    for name, collection in collections.items():
        results_summary[name]["evaluable"] += 1
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        metas = results["metadatas"][0]

        if metas[0]["is_selected"] == 1:
            results_summary[name]["top1"] += 1
        if any(m["is_selected"] == 1 for m in metas):
            results_summary[name]["top3"] += 1

print("\n" + "=" * 70)
print("CHUNKING STRATEGY COMPARISON")
print("=" * 70)
print(f"{'Strategy':<20} {'Chunks':<10} {'Top-1 Acc':<15} {'Top-3 Acc':<15}")
print("-" * 70)

for name in strategies:
    n_evaluable = results_summary[name]["evaluable"]
    top1 = results_summary[name]["top1"]
    top3 = results_summary[name]["top3"]
    top1_pct = 100 * top1 / n_evaluable if n_evaluable else 0
    top3_pct = 100 * top3 / n_evaluable if n_evaluable else 0
    total_chunks = collections[name].count()
    print(f"{name:<20} {total_chunks:<10} {top1}/{n_evaluable} ({top1_pct:.1f}%){'':<3} {top3}/{n_evaluable} ({top3_pct:.1f}%)")

print("\nDone. Use this table directly in your submission's chunking comparison section.")
