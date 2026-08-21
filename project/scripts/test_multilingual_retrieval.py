import csv
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================================================
# FILE
# =========================================================

CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5


# =========================================================
# LOAD DATA
# =========================================================

print("==========================================")
print("MULTILINGUAL RETRIEVAL TEST")
print("==========================================")

print("\nReading:")
print(CSV_FILE)


rows = []

with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        rows.append(row)


print("\nTotal rows:", len(rows))


# =========================================================
# FIND QUERY
# =========================================================

QUERY_ID = "1102432"

query_rows = [
    row for row in rows
    if row["query_id"] == QUERY_ID
]


print(
    "Rows for query:",
    len(query_rows)
)


if len(query_rows) == 0:

    print("ERROR: Query not found.")

    raise SystemExit


query = query_rows[0]["query"]


print("\nQuery:")
print(query)


# =========================================================
# SHOW CORRECT ANSWER
# =========================================================

correct_rows = [
    row for row in query_rows
    if row["is_selected"] == "1"
]


print(
    "\nCorrect passages:",
    len(correct_rows)
)


for row in correct_rows:

    print("\nEXPECTED PASSAGE ID:")
    print(row["passage_id"])

    print("\nEXPECTED PASSAGE:")
    print(row["passage"])


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading multilingual embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

passage_texts = [
    row["passage"]
    for row in query_rows
]


print("\nCreating embeddings...")


passage_embeddings = model.encode(
    passage_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)


# Normalize

passage_embeddings = (
    passage_embeddings /
    np.linalg.norm(
        passage_embeddings,
        axis=1,
        keepdims=True
    )
)


# =========================================================
# QUERY EMBEDDING
# =========================================================

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)[0]


query_embedding = (
    query_embedding /
    np.linalg.norm(query_embedding)
)


# =========================================================
# SIMILARITY
# =========================================================

similarities = (
    passage_embeddings @
    query_embedding
)


ranking = np.argsort(
    similarities
)[::-1]


# =========================================================
# RESULTS
# =========================================================

print("\n==========================================")
print("TOP RESULTS")
print("==========================================")


correct_id = correct_rows[0]["passage_id"]


correct_rank = None


for rank, index in enumerate(
    ranking[:TOP_K],
    start=1
):

    row = query_rows[index]

    passage_id = row["passage_id"]

    similarity = similarities[index]


    if passage_id == correct_id:

        correct_rank = rank


    print(
        f"\nRank: {rank}"
    )

    print(
        f"Passage ID: {passage_id}"
    )

    print(
        f"Similarity: {similarity:.4f}"
    )

    print(
        "Selected:",
        row["is_selected"]
    )

    print(
        "Passage:"
    )

    print(
        row["passage"]
    )

    print("------------------------------------------")


# =========================================================
# FINAL RESULT
# =========================================================

print("\n==========================================")
print("FINAL RESULT")
print("==========================================")


if correct_rank is not None:

    print(
        "Correct passage rank:",
        correct_rank
    )

else:

    print(
        "Correct passage is NOT in Top-5"
    )


if correct_rank is not None:

    print(
        "Top-5 Retrieval: CORRECT"
    )

else:

    print(
        "Top-5 Retrieval: WRONG"
    )