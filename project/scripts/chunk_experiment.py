import csv
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import chunk_fixed


PASSAGES_FILE = "passages.csv"
QUERIES_FILE = "test_queries.csv"

CHUNK_SIZES = [50, 100, 200]
TOP_K = 5


print("==========================================")
print("CHUNK + EMBEDDING EXPERIMENT")
print("==========================================")


# ---------------------------------------------------------
# LOAD ORIGINAL PASSAGES
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


print("Original passages:", len(passages))


# ---------------------------------------------------------
# LOAD TEST QUERIES
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


print("Test queries:", len(queries))


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ---------------------------------------------------------
# RUN EXPERIMENT FOR EACH CHUNK SIZE
# ---------------------------------------------------------

for chunk_size in CHUNK_SIZES:

    print("\n")
    print("==========================================")
    print("CHUNK SIZE:", chunk_size)
    print("==========================================")


    # Create chunks

    chunks = []

    for passage in passages:

        passage_chunks = chunk_fixed(
            passage["text"],
            chunk_size=chunk_size,
            overlap=0
        )

        for index, chunk in enumerate(
            passage_chunks
        ):

            chunks.append({
                "passage_id": passage["passage_id"],
                "chunk_id": index,
                "text": chunk
            })


    print("Chunks created:", len(chunks))


    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )


    # Normalize

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = embeddings / norms


    # -----------------------------------------------------
    # EVALUATE QUERIES
    # -----------------------------------------------------

    correct = 0


    for number, item in enumerate(
        queries,
        start=1
    ):

        query = item["query"]

        expected_passage_id = (
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


        # Top K chunks

        top_indices = np.argsort(
            similarities
        )[::-1][:TOP_K]


        retrieved_passage_ids = []

        for index in top_indices:

            retrieved_passage_ids.append(
                chunks[index]["passage_id"]
            )


        found = (
            expected_passage_id
            in retrieved_passage_ids
        )


        if found:

            correct += 1


        print(
            f"Query {number}: "
            f"{'CORRECT' if found else 'WRONG'}"
        )


    # -----------------------------------------------------
    # ACCURACY
    # -----------------------------------------------------

    total = len(queries)


    if total > 0:

        accuracy = (
            correct / total
        ) * 100

    else:

        accuracy = 0


    print("\nRESULT")

    print(
        "Chunk size:",
        chunk_size
    )

    print(
        "Correct:",
        correct
    )

    print(
        "Total:",
        total
    )

    print(
        "Top-5 Accuracy:",
        round(accuracy, 2),
        "%"
    )


print("\n")
print("==========================================")
print("EXPERIMENT COMPLETED")
print("==========================================")
