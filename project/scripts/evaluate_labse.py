import csv
import numpy as np
from sentence_transformers import SentenceTransformer


CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

MODEL_NAME = "sentence-transformers/LaBSE"


# =========================================================
# LOAD DATA
# =========================================================

print("==========================================")
print("FULL RETRIEVAL EVALUATION")
print("==========================================")

print("\nLoading dataset...")

with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)
    rows = list(reader)


# =========================================================
# FIND RELEVANT QUERIES
# =========================================================

query_ids = []

for row in rows:

    if row["is_selected"] == "1":

        if row["query_id"] not in query_ids:

            query_ids.append(row["query_id"])


print("\nRelevant unique queries:", len(query_ids))


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading model:")

print(MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# =========================================================
# METRICS
# =========================================================

recall_1 = 0
recall_3 = 0
recall_5 = 0

reciprocal_ranks = []


# =========================================================
# EVALUATE EACH QUERY
# =========================================================

for number, query_id in enumerate(
    query_ids,
    start=1
):

    query_rows = [
        row for row in rows
        if row["query_id"] == query_id
    ]


    if not query_rows:
        continue


    query = query_rows[0]["query"]


    correct_ids = {
        row["passage_id"]
        for row in query_rows
        if row["is_selected"] == "1"
    }


    passages = [
        row["passage"]
        for row in query_rows
    ]


    # -----------------------------------------------------
    # EMBEDDINGS
    # -----------------------------------------------------

    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        show_progress_bar=False
    )


    passage_embeddings = (
        passage_embeddings /
        np.linalg.norm(
            passage_embeddings,
            axis=1,
            keepdims=True
        )
    )


    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]


    query_embedding = (
        query_embedding /
        np.linalg.norm(query_embedding)
    )


    # -----------------------------------------------------
    # SIMILARITY
    # -----------------------------------------------------

    similarities = (
        passage_embeddings @
        query_embedding
    )


    ranking = np.argsort(
        similarities
    )[::-1]


    ranked_ids = [
        query_rows[index]["passage_id"]
        for index in ranking
    ]


    # -----------------------------------------------------
    # FIND FIRST RELEVANT PASSAGE
    # -----------------------------------------------------

    first_relevant_rank = None


    for rank, passage_id in enumerate(
        ranked_ids,
        start=1
    ):

        if passage_id in correct_ids:

            first_relevant_rank = rank
            break


    # -----------------------------------------------------
    # RECALL
    # -----------------------------------------------------

    if first_relevant_rank == 1:

        recall_1 += 1


    if first_relevant_rank is not None and first_relevant_rank <= 3:

        recall_3 += 1


    if first_relevant_rank is not None and first_relevant_rank <= 5:

        recall_5 += 1


    # -----------------------------------------------------
    # MRR
    # -----------------------------------------------------

    if first_relevant_rank is not None:

        reciprocal_ranks.append(
            1 / first_relevant_rank
        )

    else:

        reciprocal_ranks.append(0)


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    print(
        f"\nQuery {number}: {query}"
    )


    if first_relevant_rank is not None:

        print(
            "Correct passage rank:",
            first_relevant_rank
        )

    else:

        print(
            "Correct passage not found"
        )


# =========================================================
# FINAL METRICS
# =========================================================

total = len(query_ids)


recall_1_percent = (
    recall_1 / total * 100
)


recall_3_percent = (
    recall_3 / total * 100
)


recall_5_percent = (
    recall_5 / total * 100
)


mrr = np.mean(
    reciprocal_ranks
)


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n")
print("==========================================")
print("FINAL RESULTS")
print("==========================================")

print(
    "Model:",
    MODEL_NAME
)

print(
    "Total queries:",
    total
)

print(
    f"Recall@1: {recall_1_percent:.2f} %"
)

print(
    f"Recall@3: {recall_3_percent:.2f} %"
)

print(
    f"Recall@5: {recall_5_percent:.2f} %"
)

print(
    f"MRR: {mrr:.4f}"
)

print("==========================================")