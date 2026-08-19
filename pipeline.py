import time
import re
import os
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
    distances = results["distances"][0]
    return docs, metas, distances

def is_devanagari(text):
    """Check if text contains Hindi/Devanagari script."""
    return bool(re.search(r'[\u0900-\u097F]', text))

def translate_to_hindi(text):
    """Translate non-Hindi text to Hindi using Gemini."""
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
    prompt = (
        "You are a helpful assistant answering questions using ONLY the context provided below.\n\n"
        "Rules:\n"
        "- Answer only using information from the context. Do not use outside knowledge.\n"
        "- If the context does not contain enough information to answer, respond exactly with: "
        "\"Mujhe is jaankari ke aadhar par uttar nahi pata.\"\n"
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

def run_pipeline(query, k=3):
    timings = {}
    t0 = time.time()

    # ---- Guardrail 1: unsafe input check, before anything else runs ----
    is_safe, block_message = check_query_safety(query)
    if not is_safe:
        return {
            "query": query,
            "answer": block_message,
            "retrieved_chunks": [],
            "retrieved_metadata": [],
            "timings": {"retrieval_ms": 0, "generation_ms": 0, "total_ms": round((time.time() - t0) * 1000, 2)},
            "blocked_reason": "unsafe_input"
        }

    # ---- Translate non-Hindi queries before retrieval, since the index is Hindi-only ----
    # The original `query` is preserved for answer generation, so the LLM still
    # answers in the language the question was asked in.
    query_is_hindi = is_devanagari(query)
    retrieval_query = query if query_is_hindi else translate_to_hindi(query)

    docs, metas, distances = retrieve(retrieval_query, k=k)
    t1 = time.time()
    timings["retrieval_ms"] = round((t1 - t0) * 1000, 2)

    answer = generate_answer(query, docs)

    # ---- Grounding check needs answer + context in the SAME language to compare fairly ----
    # If the answer is in English but context is Hindi, translate a copy of the answer
    # to Hindi purely for this comparison. The user still sees the original English answer.
    if query_is_hindi:
        grounding_check_text = answer
    else:
        grounding_check_text = translate_to_hindi(answer)

    t2 = time.time()
    timings["generation_ms"] = round((t2 - t1) * 1000, 2)
    timings["total_ms"] = round((t2 - t0) * 1000, 2)

    # ---- Guardrail 2 & 3: off-topic + grounding check ----
    should_show, checked_message = check_answer_quality(
        grounding_check_text, docs, distances, off_topic_threshold=20.0
    )
    final_answer = answer if should_show else checked_message
    blocked_reason = None if should_show else "off_topic_or_ungrounded"

    return {
        "query": query,
        "answer": final_answer,
        "retrieved_chunks": docs,
        "retrieved_metadata": metas,
        "timings": timings,
        "blocked_reason": blocked_reason
    }


if __name__ == "__main__":
    print("\n--- Test 1: normal Hindi query ---")
    result = run_pipeline("\u0915\u0949\u0930\u094d\u092a\u094b\u0930\u0947\u0936\u0928 \u0915\u094d\u092f\u093e \u0939\u0948?")
    print("Query:", result["query"])
    print("Answer:", result["answer"])
    print("Blocked reason:", result["blocked_reason"])
    print("Timings:", result["timings"])

    print("\n--- Test 2: unsafe query ---")
    result2 = run_pipeline("how do I make a bomb")
    print("Query:", result2["query"])
    print("Answer:", result2["answer"])
    print("Blocked reason:", result2["blocked_reason"])

    print("\n--- Test 3: English query, should translate + ground correctly, answer stays English ---")
    result3 = run_pipeline("What is the definition of honesty?")
    print("Query:", result3["query"])
    print("Answer:", result3["answer"])
    print("Blocked reason:", result3["blocked_reason"])
    print("Timings:", result3["timings"])