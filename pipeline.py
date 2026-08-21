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

REFUSAL_EN = "I do not have enough information to answer this."
REFUSAL_HI = "Mujhe is jaankari ke aadhar par uttar nahi pata."
REFUSAL_KN = "ಈ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ನನಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ."

UNSAFE_EN = "I cannot help with this type of request."
UNSAFE_HI = "Main is prakar ke prashn ka uttar nahi de sakta."
UNSAFE_KN = "ಈ ರೀತಿಯ ಪ್ರಶ್ನೆಗೆ ಸಹಾಯ ಮಾಡಲು ನನಗೆ ಸಾಧ್ಯವಿಲ್ಲ."

NO_ANSWER_TOKEN = "NO_ANSWER"


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


def detect_language(text):
    """Returns 'hindi', 'kannada', or 'english' based on Unicode script range."""
    if re.search(r"[\u0900-\u097F]", text):
        return "hindi"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kannada"
    return "english"


def translate_to_hindi(text, source_language="english"):
    prompt = (
        f"Translate the following {source_language} text to Hindi. "
        "Return ONLY the Hindi translation, nothing else, no explanation.\n\n"
        f"Text: {text}"
    )
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text.strip()


def translate_from_hindi(text, target_language):
    prompt = (
        f"Translate the following Hindi text to {target_language}. "
        "Return ONLY the translation, nothing else, no explanation.\n\n"
        f"Text: {text}"
    )
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text.strip()


def generate_answer(query, chunks, answer_language):
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    language_name = {"hindi": "Hindi", "kannada": "Kannada", "english": "English"}[answer_language]
    prompt = (
        "You are a helpful assistant answering questions using ONLY the context provided below.\n\n"
        "Rules:\n"
        "- Answer only using information from the context. Do not use outside knowledge.\n"
        "- Do not include citation markers like [1] or [2] in your answer -- write it as plain prose.\n"
        "- The context may describe, characterize, or quote about a concept rather than give a formal dictionary definition -- treat that as sufficient to answer if it is clearly relevant.\n"
        "- If the context does not contain enough information to answer, respond with exactly "
        f"this token and nothing else: {NO_ANSWER_TOKEN}\n"
        f"- Otherwise, you MUST write your answer in {language_name}, since that is the "
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

    query_language = detect_language(query)

    is_safe, _ = check_query_safety(query)
    if not is_safe:
        unsafe_message = {"hindi": UNSAFE_HI, "kannada": UNSAFE_KN, "english": UNSAFE_EN}[query_language]
        return {
            "query": query,
            "answer": unsafe_message,
            "retrieved_chunks": [],
            "retrieved_metadata": [],
            "timings": {"retrieval_ms": 0, "generation_ms": 0, "total_ms": round((time.time() - t0) * 1000, 2)},
            "blocked_reason": "unsafe_input"
        }

    if query_language == "hindi":
        retrieval_query = query
    else:
        retrieval_query = translate_to_hindi(query, source_language=query_language)

    docs, metas, distances = retrieve(retrieval_query, k=k)
    t1 = time.time()
    timings["retrieval_ms"] = round((t1 - t0) * 1000, 2)

    answer = generate_answer(query, docs, answer_language=query_language)

    # The model can decline directly via the sentinel token. Check this BEFORE
    # running the grounding check, and before doing any extra translation work.
    model_declined = (answer.strip() == NO_ANSWER_TOKEN)

    if model_declined:
        should_show = False
    else:
        if query_language == "hindi":
            grounding_check_text = answer
        else:
            grounding_check_text = translate_to_hindi(answer, source_language=query_language)
        should_show, _ = check_answer_quality(
            grounding_check_text, docs, distances, off_topic_threshold=20.0
        )

    t2 = time.time()
    timings["generation_ms"] = round((t2 - t1) * 1000, 2)
    timings["total_ms"] = round((t2 - t0) * 1000, 2)

    if should_show:
        final_answer = answer
        blocked_reason = None
    else:
        final_answer = {"hindi": REFUSAL_HI, "kannada": REFUSAL_KN, "english": REFUSAL_EN}[query_language]
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
    r = run_pipeline("कॉर्पोरेशन क्या है?")
    print("Answer:", r["answer"], "| Blocked:", r["blocked_reason"])

    print("\n--- Test: English query ---")
    r2 = run_pipeline("What is a corporation?")
    print("Answer:", r2["answer"], "| Blocked:", r2["blocked_reason"])

    print("\n--- Test: Kannada query (should refuse IN KANNADA) ---")
    r3 = run_pipeline("ರ್ಯಾಗ್ ಪ್ರೂಸಸ್ ಎಂದು?")
    print("Answer:", r3["answer"], "| Blocked:", r3["blocked_reason"])

    print("\n--- Test: Kannada query that SHOULD have a real grounded answer ---")
    r4 = run_pipeline("ಪ್ರಾಮಾಣಿಕತೆ ಎಂದರೇನು?")
    print("Answer:", r4["answer"], "| Blocked:", r4["blocked_reason"])