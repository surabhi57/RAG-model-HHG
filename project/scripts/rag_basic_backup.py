import os

print("RUNNING FILE:")
print(os.path.abspath(__file__))

PASSAGES_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

print("CSV FILE:")
print(PASSAGES_FILE)
import csv
import numpy as np
from sentence_transformers import SentenceTransformer


import os

PASSAGES_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

print("Reading passages from:")
print(os.path.abspath(PASSAGES_FILE))

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5


print("==========================================")
print("BASIC RAG RETRIEVAL SYSTEM")
print("==========================================")


# Load passages

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
                "passage_id": row["passage_id"],
                "text": text
            })


print(
    "Passages loaded:",
    len(passages)
)
print("\nChecking corporation passage...")

for item in passages:
    if item["passage_id"] == "5":
        print("Passage ID 5:")
        print(item["text"][:500])
        break

# Load model

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Model loaded.")


# Create embeddings

print("\nCreating passage embeddings...")

passage_texts = [
    item["text"]
    for item in passages
]

embeddings = model.encode(
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


print("Embeddings created.")


# Query loop

print("\n==========================================")
print("RAG SYSTEM READY")
print("==========================================")


while True:

    query = input(
        "\nEnter your question "
        "(type 'exit' to stop): "
    )


    if query.lower() == "exit":

        print("RAG system stopped.")
        break


    # Query embedding

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]


    query_embedding = (
        query_embedding /
        np.linalg.norm(query_embedding)
    )


    # Similarity

    similarities = (
        embeddings @
        query_embedding
    )


    # Ranking

    ranked_indices = np.argsort(
        similarities
    )[::-1]


    # Display results

    print("\n==========================================")
    print("TOP", TOP_K, "RETRIEVED PASSAGES")
    print("==========================================")


    for rank, index in enumerate(
        ranked_indices[:TOP_K],
        start=1
    ):

        print(
            "\nRank:",
            rank
        )

        print(
            "Passage ID:",
            passages[index]["passage_id"]
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

        print(
            "Passage:"
        )

        print(
            passages[index]["text"]
        )

        print(
            "------------------------------------------"
        )