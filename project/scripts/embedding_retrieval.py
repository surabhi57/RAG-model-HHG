import csv
import numpy as np
from sentence_transformers import SentenceTransformer


PASSAGES_FILE = "passages.csv"
QUERIES_FILE = "test_queries.csv"

TOP_K = 5


print("==========================================")
print("EMBEDDING RETRIEVAL STARTED")
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


print("Total passages loaded:", len(passages))


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


print("Total queries loaded:", len(queries))


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


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

print("Passage embeddings created.")


# ---------------------------------------------------------
# RETRIEVE PASSAGES
# ---------------------------------------------------------

def retrieve(query, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]

    # Normalize embeddings
    query_embedding = (
        query_embedding /
        np.linalg.norm(query_embedding)
    )

    normalized_passages = (
        passage_embeddings /
        np.linalg.norm(
            passage_embeddings,
            axis=1,
            keepdims=True
        )
    )

    # Cosine similarity
    similarities = (
        normalized_passages
        @ query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            (
                similarities[index],
                passages[index]
            )
        )

    return results


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------

correct = 0

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

    results = retrieve(
        query,
        TOP_K
    )

    retrieved_ids = []

    for score, passage in results:

        retrieved_ids.append(
            passage["passage_id"]
        )

    found = expected_id in retrieved_ids

    if found:
        correct += 1

    print("\n------------------------------------------")

    print("Query", number)

    print("Query:", query)

    print(
        "Expected passage ID:",
        expected_id
    )

    print(
        "Retrieved passage IDs:",
        retrieved_ids
    )

    print(
        "Correct:",
        found
    )


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

total = len(queries)

if total > 0:

    accuracy = (
        correct / total
    ) * 100

else:

    accuracy = 0


print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

print("Correct:", correct)

print("Total:", total)

print(
    "Top-5 Semantic Retrieval Accuracy:",
    round(accuracy, 2),
    "%"
)

print("==========================================")