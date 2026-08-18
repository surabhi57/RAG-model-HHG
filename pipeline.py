import time
import os
import pandas as pd
import chromadb
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

from chunking import chunk_fixed

load_dotenv()

print("Setting up pipeline (this happens once)...")

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def build_index(sample_size=100):
    path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename="validation/hinval.parquet",
        repo_type="dataset"
    )
    df = pd.read_parquet(path)
    df_sample = df.iloc[:sample_size]

    records = []
    for _, row in df_sample.iterrows():
        passages_dict = row["passages"]
        hindi_passages = passages_dict["Translated_passages"]
        is_selected = passages_dict["is_selected"]
        for idx, (passage_text, selected_flag) in enumerate(zip(hindi_passages, is_selected)):
            records.append({
                "text": passage_text,
                "query_id": int(row["query_id"]),
                "query": row["query"],
                "passage_idx": idx,
                "is_selected": int(selected_flag)
            })

    chunked_records = []
    for rec in records:
        chunks = chunk_fixed(rec["text"], chunk_size=80, overlap=20)
        for c_idx, chunk_text in enumerate(chunks):
            chunked_records.append({
                "chunk_text": chunk_text,
                "query_id": rec["query_id"],
                "query": rec["query"],
                "passage_idx": rec["passage_idx"],
                "is_selected": rec["is_selected"],
                "chunk_idx": c_idx
            })

    texts = [r["chunk_text"] for r in chunked_records]
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()

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

    collection.add(embeddings=embeddings, documents=texts, ids=ids, metadatas=metadatas)
    print(f"Index built: {len(chunked_records)} chunks from {sample_size} queries.")
    return collection

collection = build_index(sample_size=100)

def retrieve(query, k=3):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas

def generate_answer(query, chunks):
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    prompt = (
        "You are a helpful assistant answering questions using ONLY the context provided below.\n\n"
        "Rules:\n"
        "- Answer only using information from the context. Do not use outside knowledge.\n"
        "- If the context does not contain enough information to answer, respond exactly with: "
        "\"Mujhe is jaankari ke aadhar par uttar nahi pata.\"\n"
        "- Keep the answer concise and in the same language as the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text.strip()

def run_pipeline(query, k=3):
    timings = {}
    t0 = time.time()

    docs, metas = retrieve(query, k=k)
    t1 = time.time()
    timings["retrieval_ms"] = round((t1 - t0) * 1000, 2)

    answer = generate_answer(query, docs)
    t2 = time.time()
    timings["generation_ms"] = round((t2 - t1) * 1000, 2)

    timings["total_ms"] = round((t2 - t0) * 1000, 2)

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": docs,
        "retrieved_metadata": metas,
        "timings": timings
    }


if __name__ == "__main__":
    test_query = "\u0915\u0949\u0930\u094d\u092a\u094b\u0930\u0947\u0936\u0928 \u0915\u094d\u092f\u093e \u0939\u0948?"
    result = run_pipeline(test_query)

    print("\n=== PIPELINE RESULT ===")
    print("Query:", result["query"])
    print("Answer:", result["answer"])
    print("Timings:", result["timings"])
    print("\nTop retrieved chunk (is_selected =", result["retrieved_metadata"][0]["is_selected"], "):")
    print(result["retrieved_chunks"][0][:200])
