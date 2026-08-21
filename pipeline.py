import time
import re
import os
import json
import pandas as pd
import chromadb
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

from chunking import chunk_fixed
from guardrails import check_query_safety, check_answer_quality

load_dotenv()

print("Setting up pipeline (this happens once)...")

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

REFUSAL_EN = "I do not have enough information to answer this."
REFUSAL_HI = "Mujhe is jaankari ke aadhar par uttar nahi pata."
UNSAFE_EN = "I cannot help with this type of request."
UNSAFE_HI = "Main is prakar ke prashn ka uttar nahi de sakta."

CACHE_DIR = "chroma_storage"
CACHE_MARKER = os.path.join(CACHE_DIR, "index_marker.json")

def build_index(sample_size=100):
    os.makedirs(CACHE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CACHE_DIR)

    if os.path.exists(CACHE_MARKER):
        with open(CACHE_MARKER, "r") as f:
            marker = json.load(f)
        if marker.get("sample_size") == sample_size:
            try:
                collection = client.get_collection("msmarco_chunks")
                if collection.count() == marker.get("chunk_count"):
                    print(f"Loaded cached index: {collection.count()} chunks from {sample_size} queries (skipped rebuild).")
                    return collection
            except Exception:
                pass

    print("No valid cache found, building index from scratch...")

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

    BATCH_SIZE = 4000
    for i in range(0, len(ids), BATCH_SIZE):
        collection.add(
            embeddings=embeddings[i:i+BATCH_SIZE],
            documents=texts[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE]
        )

    with open(CACHE_MARKER, "w") as f:
        json.dump({"sample_size": sample_size, "chunk_count": len(chunked_records)}, f)

    print(f"Index built: {len(chunked_records)} chunks from {sample_size} queries.")
    return collection

collection = build_index(sample_size=100)

def retrieve(query, k=6):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return docs, metas, distances

def is_devanagari(text):
    return bool(re.search(r"[\u0900-\u097F]", text))

def translate_to_hindi(text):
    prompt = (
        "Translate the following text to Hindi. "
        "Return ONLY the Hindi translation, nothing else, no explanation.\n\n"
        f"Text: {text}"
    )
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text.strip()

def generate_answer(query, chunks):
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    answer_language = "Hindi" if is_devanagari(query) else "English"
    refusal_line = (
        'If the context does not contain enough information to answer, respond exactly with: '
        '"Mujhe is jaankari ke aadhar par uttar nahi pata."'
        if answer_language == "Hindi" else
        'If the context does not contain enough information to answer, respond exactly with: '
        '"I do not have enough information to answer this."'
    )
    prompt = (
        "You are a helpful assistant answering questions using ONLY the context provided below.\n\n"
        "Rules:\n"
        "- Answer only using information from the context. Do not use outside knowledge.\n"
        f"- {refusal_line}\n"
        f"- The context below may be in Hindi even if the question is in English. Regardless of the "
        f"context's language, you MUST write your answer in {answer_language}, since that is the "
        "language the question was asked in.\n"
        "- Keep the answer concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text.strip()

def run_pipeline(query, k=6):
    timings = {}
    t0 = time.time()

    query_is_hindi = is_devanagari(query)

    is_safe, _ = check_query_safety(query)
    if not is_safe:
        return {
            "query": query,
            "answer": UNSAFE_HI if query_is_hindi else UNSAFE_EN,
            "retrieved_chunks": [],
            "retrieved_metadata": [],
            "timings": {"retrieval_ms": 0, "generation_ms": 0, "total_ms": round((time.time() - t0) * 1000, 2)},
            "blocked_reason": "unsafe_input"
        }

    retrieval_query = query if query_is_hindi else translate_to_hindi(query)

    docs, metas, distances = retrieve(retrieval_query, k=k)
    t1 = time.time()
    timings["retrieval_ms"] = round((t1 - t0) * 1000, 2)

    answer = generate_answer(query, docs)

    grounding_check_text = answer if query_is_hindi else translate_to_hindi(answer)

    t2 = time.time()
    timings["generation_ms"] = round((t2 - t1) * 1000, 2)
    timings["total_ms"] = round((t2 - t0) * 1000, 2)

    should_show, _ = check_answer_quality(
        grounding_check_text, docs, distances, off_topic_threshold=20.0
    )

    if should_show:
        final_answer = answer
        blocked_reason = None
    else:
        final_answer = REFUSAL_HI if query_is_hindi else REFUSAL_EN
        blocked_reason = "off_topic_or_ungrounded"

    return {
        "query": query,
        "answer": final_answer,
        "retrieved_chunks": docs,
        "retrieved_metadata": metas,
        "timings": timings,
        "blocked_reason": blocked_reason
    }


if __name__ == "__main__":
    print("\n--- Test: Hindi query ---")
    result = run_pipeline("\u0915\u0949\u0930\u094d\u092a\u094b\u0930\u0947\u0936\u0928 \u0915\u094d\u092f\u093e \u0939\u0948?")
    print("Answer:", result["answer"], "| Blocked:", result["blocked_reason"])