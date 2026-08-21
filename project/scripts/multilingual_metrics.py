import csv
import numpy as np
from sentence_transformers import SentenceTransformer


PASSAGES_FILE = "passages.csv"
QUERIES_FILE = "test_queries.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


print("==========================================")
print("MULTILINGUAL RETRIEVAL EVALUATION")
print("==========================================")


# ---------------------------------------------------------
# LOAD PASSAGES
# ---------------------------------------------------------

passages = []

print("\nReading passages.csv...")

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


print("Passages loaded:", len(passages))


# ---------------------------------------------------------
# LOAD QUERIES
# ---------------------------------------------------------

queries = []

print("\nReading test_queries.csv...")

with open(
    QUERIES_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        queries.append(row)


print("Queries loaded:", len(queries))


# ---------------------------------------------------------
# LOAD MULTILINGUAL MODEL
# ---------------------------------------------------------

print("\nLoading multilingual model...")

print(
    "Model:",
    MODEL_NAME
)


model = SentenceTransformer(
    MODEL_NAME
)


print("Multilingual model loaded successfully.")


# ---------------------------------------------------------
# CREATE PASSAGE EMBEDDINGS
# ---------------------------------------------------------

print("\nCreating passage embeddings...")

passage_texts = [
    passage["text"]
    for passage in passages
]


passage_embeddings = model.encode(
    passage_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)


# Normalize embeddings

passage_embeddings = (
    passage_embeddings /
    np.linalg.norm(
        passage_embeddings,
        axis=1,
        keepdims=True
    )
)


print("Passage embeddings created.")


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0

reciprocal_ranks = []


# ---------------------------------------------------------
# EVALUATE EACH QUERY
# ---------------------------------------------------------

print("\n==========================================")
print("QUERY RESULTS")
print("==========================================")


for number, item in enumerate(
    queries,
    start=1
):

    query = item["query"]

    expected_id = (
        item["expected_passage_id"]
    )


    # -----------------------------------------------------
    # CREATE QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]


    query_embedding = (
        query_embedding /
        np.linalg.norm(query_embedding)
    )


    # -----------------------------------------------------
    # CALCULATE SIMILARITY
    # -----------------------------------------------------

    similarities = (
        passage_embeddings @
        query_embedding
    )


    # Sort highest similarity first

    ranked_indices = np.argsort(
        similarities
    )[::-1]


    ranked_ids = [
        passages[index]["passage_id"]
        for index in ranked_indices
    ]


    # -----------------------------------------------------
    # RECALL@1
    # -----------------------------------------------------

    if expected_id in ranked_ids[:1]:

        recall_at_1 += 1


    # -----------------------------------------------------
    # RECALL@3
    # -----------------------------------------------------

    if expected_id in ranked_ids[:3]:

        recall_at_3 += 1


    # -----------------------------------------------------
    # RECALL@5
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

        reciprocal_rank = 1 / rank

        reciprocal_ranks.append(
            reciprocal_rank
        )

        print(
            "Query",
            number,
            "-> rank:",
            rank
        )

    else:

        reciprocal_ranks.append(0)

        print(
            "Query",
            number,
            "-> not found"
        )


# ---------------------------------------------------------
# CALCULATE FINAL METRICS
# ---------------------------------------------------------

total = len(queries)


if total > 0:

    recall1 = (
        recall_at_1 /
        total
    ) * 100


    recall3 = (
        recall_at_3 /
        total
    ) * 100


    recall5 = (
        recall_at_5 /
        total
    ) * 100


    mrr = np.mean(
        reciprocal_ranks
    )


else:

    recall1 = 0
    recall3 = 0
    recall5 = 0
    mrr = 0


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

print("\n==========================================")
print("FINAL MULTILINGUAL RESULTS")
print("==========================================")


print(
    "Recall@1:",
    round(recall1, 2),
    "%"
)


print(
    "Recall@3:",
    round(recall3, 2),
    "%"
)


print(
    "Recall@5:",
    round(recall5, 2),
    "%"
)


print(
    "MRR:",
    round(mrr, 4)
)


print("==========================================")