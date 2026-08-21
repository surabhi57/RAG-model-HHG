print(">>> THIS IS RAG_FINAL.PY <<<")
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\bilingual_passages.csv"

EMBEDDING_MODEL = "sentence-transformers/LaBSE"

LLM_MODEL = "google/flan-t5-small"

TOP_K = 5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL RAG SYSTEM")
print("=" * 70)

print("\nReading bilingual dataset...")

df = pd.read_csv(
    CSV_FILE,
    encoding="utf-8"
)

print("Total rows:", len(df))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading LaBSE...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("LaBSE loaded.")


# ============================================================
# CREATE ENGLISH PASSAGE EMBEDDINGS
# ============================================================

print("\nCreating English passage embeddings...")

english_passages = (
    df["english_passage"]
    .fillna("")
    .astype(str)
    .tolist()
)

passage_embeddings = embedding_model.encode(
    english_passages,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embeddings created.")


# ============================================================
# LOAD LLM
# ============================================================

print("\nLoading FLAN-T5...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

llm = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)

print("FLAN-T5 loaded.")


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(english_query, top_k=5):

    query_embedding = embedding_model.encode(
        [english_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    similarities = np.dot(
        passage_embeddings,
        query_embedding
    )

    ranking = np.argsort(
        similarities
    )[::-1]

    results = []

    for rank, index in enumerate(
        ranking[:top_k],
        start=1
    ):

        results.append({
            "rank": rank,
            "passage_id": df.iloc[index]["passage_id"],
            "similarity": float(similarities[index]),
            "english_passage": df.iloc[index]["english_passage"],
            "assamese_passage": df.iloc[index]["assamese_passage"]
        })

    return results


# ============================================================
# GENERATION
# ============================================================

def generate_answer(question, retrieved):

    # Use ONLY the highest-ranked passage
    context = retrieved[1]["english_passage"]

    prompt = (
        "Question: " + question + "\n"
        "Context: " + context + "\n"
        "Answer:"
    )

    print("\nSending prompt to FLAN-T5...")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )

    print("Generating...")

    with torch.no_grad():

        outputs = llm.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,
            num_beams=2,
            do_sample=False
        )

    print("Generation completed.")

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if not answer:
        answer = "I don't know"

    return answer

    # Use English passages for generation
    context = "\n\n".join(
        item["english_passage"]
        for item in retrieved
    )

    prompt = (
        "Answer the question using only the information "
        "provided in the context.\n\n"
        "Question: " + question + "\n\n"
        "Context:\n" + context + "\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = llm.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            do_sample=False
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# SYSTEM READY
# ============================================================

print("\n")
print("=" * 70)
print("RAG SYSTEM READY")
print("=" * 70)

print("\nFor this test, enter the English query.")
print("Type 'exit' to stop.")


while True:

    english_query = input(
        "\nEnglish Question: "
    ).strip()

    if english_query.lower() == "exit":
        break

    if not english_query:
        continue


    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    retrieved = retrieve(
        english_query,
        TOP_K
    )


    # --------------------------------------------------------
    # SHOW RETRIEVAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP 5 RETRIEVED ENGLISH PASSAGES")
    print("=" * 70)

    for item in retrieved:

        print("\nRank:", item["rank"])
        print("Passage ID:", item["passage_id"])
        print(
            "Similarity:",
            round(item["similarity"], 4)
        )

        print("Passage:")
        print(item["english_passage"])


    # --------------------------------------------------------
    