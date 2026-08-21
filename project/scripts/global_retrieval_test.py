import csv
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

MODEL_NAME = "sentence-transformers/LaBSE"

QUERY = "কৰ্পোৰেচন কি?"

EXPECTED_QUERY_ID = "1102432"

EXPECTED_PASSAGE_ID = "5"

TOP_K = 5


# =========================================================
# LOAD DATA
# =========================================================

print("==========================================")
print("GLOBAL RETRIEVAL TEST")
print("==========================================")


print("\nReading dataset...")


with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


print(
    "Total rows:",
    len(rows)
)


# =========================================================
# CHECK EXPECTED PASSAGE
# =========================================================

expected_rows = [
    row for row in rows
    if (
        row["query_id"] == EXPECTED_QUERY_ID
        and row["passage_id"] == EXPECTED_PASSAGE_ID
    )
]


print(
    "Expected passage found:",
    len(expected_rows)
)


if len(expected_rows) == 0:

    print(
        "ERROR: Expected passage was not found."
    )

    raise SystemExit


print("\nExpected passage:")
print(
    expected_rows[0]["passage"]
)


# =========================================================
# CREATE UNIQUE PASSAGE CORPUS
# =========================================================

print("\nCreating passage corpus...")


corpus = []

seen = set()


for row in rows:

    key = (
        row["query_id"],
        row["passage_id"]
    )


    if key in seen:

        continue


    seen.add(key)


    corpus.append(
        {
            "query_id": row["query_id"],
            "passage_id": row["passage_id"],
            "text": row["passage"]
        }
    )


print(
    "Unique passages:",
    len(corpus)
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading LaBSE...")


model = SentenceTransformer(
    MODEL_NAME
)


print("Model loaded.")


# =========================================================
# CREATE PASSAGE EMBEDDINGS
# =========================================================

print("\nCreating passage embeddings...")


passages = [
    item["text"]
    for item in corpus
]


passage_embeddings = model.encode(
    passages,
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

print("\nEncoding query...")


query_embedding = model.encode(
    [QUERY],
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
# TOP RESULTS
# =========================================================

print("\n")
print("==========================================")
print("TOP 5 GLOBAL RESULTS")
print("==========================================")


correct_rank = None


for rank, index in enumerate(
    ranking[:TOP_K],
    start=1
):

    item = corpus[index]

    similarity = similarities[index]


    if (
        item["query_id"] == EXPECTED_QUERY_ID
        and item["passage_id"] == EXPECTED_PASSAGE_ID
    ):

        correct_rank = rank


    print(
        f"\nRank {rank}"
    )

    print(
        "Query ID:",
        item["query_id"]
    )

    print(
        "Passage ID:",
        item["passage_id"]
    )

    print(
        "Similarity:",
        f"{similarity:.4f}"
    )

    print(
        "Passage:"
    )

    print(
        item["text"]
    )

    print("------------------------------------------")


# =========================================================
# RESULT
# =========================================================

print("\n")
print("==========================================")
print("RESULT")
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
        "Global Top-5 Retrieval: CORRECT"
    )

else:

    print(
        "Global Top-5 Retrieval: WRONG"
    )