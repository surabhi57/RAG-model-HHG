import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

EMBEDDING_MODEL = "sentence-transformers/LaBSE"

# Small model for the first RAG generation test
LLM_MODEL = "google/flan-t5-small"

TOP_K = 5


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RAG GENERATION SYSTEM")
print("=" * 70)


# ============================================================
# STEP 1: READ DATA
# ============================================================

print("\nReading passages.csv...")

df = pd.read_csv(
    CSV_FILE,
    encoding="utf-8"
)

print("Total rows:", len(df))

df = df.drop_duplicates(
    subset=["passage_id"]
).reset_index(drop=True)

print("Unique passages:", len(df))


# ============================================================
# STEP 2: LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading LaBSE...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("LaBSE loaded successfully.")


# ============================================================
# STEP 3: CREATE PASSAGE EMBEDDINGS
# ============================================================

print("\nCreating passage embeddings...")

passages = (
    df["passage"]
    .fillna("")
    .astype(str)
    .tolist()
)

passage_embeddings = embedding_model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Passage embeddings created.")


# ============================================================
# STEP 4: LOAD LLM
# ============================================================

print("\nLoading generation model...")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

llm = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)

print("LLM loaded successfully.")


# ============================================================
# STEP 5: RETRIEVAL FUNCTION
# ============================================================

def retrieve(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        passage_embeddings
    )[0]

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
            "similarity": similarities[index],
            "passage": df.iloc[index]["passage"]
        })

    return results


# ============================================================
# STEP 6: GENERATE ANSWER
# ============================================================

def generate_answer(question, retrieved_passages):

    # Use the retrieved passages as context
    context_parts = []

    for item in retrieved_passages:
        context_parts.append(
            "Passage " + str(item["passage_id"]) + ":\n" +
            item["passage"]
        )

    context = "\n\n".join(context_parts)

    prompt = (
        "Use the context below to answer the question. "
        "Give a short answer using only the context.\n\n"
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
            max_new_tokens=80,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if answer == "":
        return "I don't know"

    return answer
    context = "\n\n".join(
        [
            item["passage"]
            for item in retrieved_passages
        ]
    )

    prompt = f"""
Answer the question using ONLY the information in the context.

If the answer cannot be found in the context, say:
I don't know.

Do not invent information.

Question:
{question}

Context:
{context}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():

        outputs = llm.generate(
            **inputs,
            max_new_tokens=100
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# STEP 7: ASK QUESTION
# ============================================================

print("\n")
print("=" * 70)
print("RAG SYSTEM READY")
print("=" * 70)

print("\nEnter your question.")
print("Type 'exit' to stop.")


while True:

    question = input("\nQuestion: ").strip()

    if question.lower() == "exit":
        print("\nRAG system stopped.")
        break

    if question == "":
        print("Please enter a question.")
        continue


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print("\nRetrieving relevant passages...")

    retrieved = retrieve(
        question,
        TOP_K
    )


    # --------------------------------------------------------
    # DISPLAY RETRIEVED PASSAGES
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP 5 RETRIEVED PASSAGES")
    print("=" * 70)

    for item in retrieved:

        print("\nRank:", item["rank"])
        print("Passage ID:", item["passage_id"])
        print(
            "Similarity:",
            round(item["similarity"], 4)
        )

        print("Passage:")
        print(item["passage"])


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATING ANSWER...")
    print("=" * 70)

    answer = generate_answer(
        question,
        retrieved
    )


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(answer)

    print("\n")