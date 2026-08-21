import csv
import numpy as np

from sentence_transformers import SentenceTransformer


CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

MODEL_NAME = "sentence-transformers/LaBSE"

QUERY = "কৰ্পোৰেচন কি?"

EXPECTED_QUERY_ID = "1102432"
EXPECTED_PASSAGE_ID = "5"


print("==========================================")
print("RETRIEVAL DIAGNOSTIC")
print("==========================================")


# =========================================================
# LOAD
# =========================================================

with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


print("Total CSV rows:", len(rows))


# =========================================================
# UNIQUE QUERY IDS
# =========================================================

query_ids = set(
    row["query_id"]
    for row in rows
)

print(
    "Unique query IDs:",
    len(query_ids)
)


# =========================================================
# UNIQUE PASSAGES
# =========================================================

passage_keys = set(
    (
        row["query_id"],
        row["passage_id"]
    )
    for row in rows
)

print(
    "Unique passages:",
    len(passage_keys)
)


# =========================================================
# FIND EXPECTED PASSAGE
# =========================================================

expected = None


for row in rows:

    if (
        row["query_id"] == EXPECTED_QUERY_ID
        and row["passage_id"] == EXPECTED_PASSAGE_ID
    ):

        expected = row["passage"]
        break


print("\nExpected passage:")

print(expected)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading LaBSE...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Model loaded.")


# =========================================================
# CREATE CORPUS
# =========================================================

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

    corpus.append(row)


texts = [
    row["passage"]
    for row in corpus
]


# =========================================================
# EMBEDDINGS
# =========================================================

print("\nCreating embeddings...")


embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)


embeddings = (
    embeddings /
    np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )
)


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

scores = embeddings @ query_embedding


ranking = np.argsort(scores)[::-1]


# =========================================================
# FIND EXPECTED PASSAGE RANK
# =========================================================

expected_index = None


for i, row in enumerate(corpus):

    if (
        row["query_id"] == EXPECTED_QUERY_ID
        and row["passage_id"] == EXPECTED_PASSAGE_ID
    ):

        expected_index = i
        break


position = np.where(
    ranking == expected_index
)[0][0]


print("\n")
print("==========================================")
print("EXPECTED PASSAGE ANALYSIS")
print("==========================================")


print(
    "Global rank:",
    int(position) + 1
)


print(
    "Similarity:",
    f"{scores[expected_index]:.4f}"
)


# =========================================================
# TOP 10
# =========================================================

print("\n")
print("==========================================")
print("TOP 10 RETRIEVED")
print("==========================================")


for rank, index in enumerate(
    ranking[:10],
    start=1
):

    row = corpus[index]

    print(
        f"\nRank {rank}"
    )

    print(
        "Query ID:",
        row["query_id"]
    )

    print(
        "Passage ID:",
        row["passage_id"]
    )

    print(
        "Similarity:",
        f"{scores[index]:.4f}"
    )

    print(
        "Text:",
        row["passage"][:300]
    )

    print("------------------------------------------")


print("\nDiagnostic completed.")