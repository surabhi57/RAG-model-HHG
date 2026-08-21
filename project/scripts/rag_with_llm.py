import csv
import numpy as np

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# =========================================================
# SETTINGS
# =========================================================

PASSAGES_FILE = "passages.csv"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = "google/flan-t5-small"

TOP_K = 5


print("==========================================")
print("RAG + LLM SYSTEM")
print("==========================================")


# =========================================================
# LOAD PASSAGES
# =========================================================

print("\nReading passages.csv...")


passages = []


with open(
    PASSAGES_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)


    for row in reader:

        text = row["passage"].strip()


        if text:

            passages.append({

                "passage_id":
                    row["passage_id"],

                "text":
                    text

            })


print(
    "Passages loaded:",
    len(passages)
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


print("Embedding model loaded.")


# =========================================================
# CREATE PASSAGE EMBEDDINGS
# =========================================================

print("\nCreating passage embeddings...")


passage_texts = [
    item["text"]
    for item in passages
]


embeddings = embedding_model.encode(
    passage_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)


# Normalize

embeddings = (
    embeddings /
    np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )
)


print("Passage embeddings created.")


# =========================================================
# LOAD LLM
# =========================================================

print("\nLoading LLM...")


tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)


llm = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)


print("LLM loaded.")


# =========================================================
# RAG LOOP
# =========================================================

print("\n==========================================")
print("RAG SYSTEM READY")
print("==========================================")


while True:

    query = input(
        "\nEnter your question "
        "(type 'exit' to stop): "
    )


    if query.lower() == "exit":

        print("System stopped.")

        break


    # =====================================================
    # QUERY EMBEDDING
    # =====================================================

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )[0]


    query_embedding = (
        query_embedding /
        np.linalg.norm(query_embedding)
    )


    # =====================================================
    # RETRIEVAL
    # =====================================================

    similarities = (
        embeddings @
        query_embedding
    )


    ranked_indices = np.argsort(
        similarities
    )[::-1]


    top_indices = ranked_indices[:TOP_K]


    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    context_parts = []


    for index in top_indices:

        context_parts.append(
            passages[index]["text"]
        )


    context = "\n\n".join(
        context_parts
    )


    # =====================================================
    # CREATE PROMPT
    # =====================================================

    prompt = f"""
Answer the question using only the information
provided in the context.

If the answer is not present in the context,
say "I don't know based on the provided context."

Context:
{context}

Question:
{query}

Answer:
"""


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    outputs = llm.generate(
        **inputs,
        max_new_tokens=100
    )


    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    print("\n==========================================")
    print("QUESTION")
    print("==========================================")

    print(query)


    print("\n==========================================")
    print("GENERATED ANSWER")
    print("==========================================")

    print(answer)


    print("\n==========================================")
    print("RETRIEVED CONTEXT")
    print("==========================================")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            passages[index]["text"]
        )

        print(
            "Similarity:",
            round(
                float(
                    similarities[index]
                ),
                4
            )
        )