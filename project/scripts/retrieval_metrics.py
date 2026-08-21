import csv
import numpy as np
from sentence_transformers import SentenceTransformer


PASSAGES_FILE = "passages.csv"
QUERIES_FILE = "test_queries.csv"

TOP_K = 5


print("==========================================")
print("RETRIEVAL METRICS")
print("==========================================")


# ---------------------------------------------------------
# LOAD PASSAGES
# ---------------------------------------------------------

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


print("Passages:", len(passages))


# ---------------------------------------------------------
# LOAD QUERIES
# ---------------------------------------------------------

queries = []

with open(
    QUERIES_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        queries.append(row)


print("Queries:", len(queries))


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded.")


# ---------------------------------------------------------
# CREATE PASSAGE EMBEDDINGS
# ---------------------------------------------------------

print("\nCreating passage embeddings...")

texts = [
    passage["text"]
    for passage in passages
]

embeddings = model.encode(
    texts,
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


# ---------------------------------------------------------
# METRIC STORAGE
# ---------------------------------------------------------

recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0

reciprocal_ranks = []


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------

for number, item in enumerate(
    queries,
    start=1
):

    query = item["query"]

    expected_id = (
        item["expected_passage_id"]
    )


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
        embeddings @ query_embedding
    )


    ranked_indices = np.argsort(
        similarities
    )[::-1]


    ranked_ids = [
        passages[index]["passage_id"]
        for index in ranked_indices
    ]


    # -----------------------------------------------------
    # Recall@1
    # -----------------------------------------------------

    if expected_id in ranked_ids[:1]:

        recall_at_1 += 1


    # -----------------------------------------------------
    # Recall@3
    # -----------------------------------------------------

    if expected_id in ranked_ids[:3]:

        recall_at_3 += 1


    # -----------------------------------------------------
    # Recall@5
    # -----------------------------------------------------

    if expected_id in ranked_ids[:5]:

        recall_at_5 += 1


    # -----------------------------------------------------
    # MRR
    # -----------------------------------------------------

    if expected_id in ranked_ids:

        rank = (
            ranked_ids.index(expected_id)
            + 1
        )

        reciprocal_ranks.append(
            1 / rank
        )

        print(
            "Query",
            number,
            ": rank =",
            rank
        )

    else:

        reciprocal_ranks.append(0)

        print(
            "Query",
            number,
            ": not found"
        )


# ---------------------------------------------------------
# FINAL METRICS
# ---------------------------------------------------------

total = len(queries)


if total > 0:

    r1 = (
        recall_at_1 /
        total
    ) * 100

    r3 = (
        recall_at_3 /
        total
    ) * 100

    r5 = (
        recall_at_5 /
        total
    ) * 100

    mrr = np.mean(
        reciprocal_ranks
    )

else:

    r1 = 0
    r3 = 0
    r5 = 0
    mrr = 0


print("\n==========================================")
print("FINAL METRICS")
print("==========================================")


print(
    "Recall@1:",
    round(r1, 2),
    "%"
)

print(
    "Recall@3:",
    round(r3, 2),
    "%"
)

print(
    "Recall@5:",
    round(r5, 2),
    "%"
)

print(
    "MRR:",
    round(mrr, 4)
)


print("==========================================")