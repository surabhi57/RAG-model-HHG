import csv
import numpy as np

from sentence_transformers import SentenceTransformer


CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

MODEL_NAME = "sentence-transformers/LaBSE"

EXPECTED_QUERY_ID = "1102432"
EXPECTED_PASSAGE_ID = "5"

ASSAMESE_QUERY = "কৰ্পোৰেচন কি?"

ENGLISH_QUERY = "what is a corporation?"


# =========================================================
# LOAD DATA
# =========================================================

with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


# =========================================================
# CREATE UNIQUE CORPUS
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
# LOAD MODEL
# =========================================================

print("Loading LaBSE...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Model loaded.")


# =========================================================
# PASSAGE EMBEDDINGS
# =========================================================

print("Creating passage embeddings...")


passage_embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)


passage_embeddings = (
    passage_embeddings /
    np.linalg.norm(
        passage_embeddings,
        axis=1,
        keepdims=True
    )
)


# =========================================================
# TEST FUNCTION
# =========================================================

def test_query(query):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]


    query_embedding = (
        query_embedding /
        np.linalg.norm(query_embedding)
    )


    similarities = (
        passage_embeddings @
        query_embedding
    )


    ranking = np.argsort(
        similarities
    )[::-1]


    expected_index = None


    for i, row in enumerate(corpus):

        if (
            row["query_id"] == EXPECTED_QUERY_ID
            and row["passage_id"] == EXPECTED_PASSAGE_ID
        ):

            expected_index = i
            break


    rank_position = np.where(
        ranking == expected_index
    )[0][0] + 1


    print("\n")
    print("==========================================")
    print("QUERY")
    print("==========================================")

    print(query)


    print(
        "\nCorrect passage rank:",
        rank_position
    )


    print(
        "Correct passage similarity:",
        f"{similarities[expected_index]:.4f}"
    )


    print("\nTOP 5:")


    for rank, index in enumerate(
        ranking[:5],
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
            f"{similarities[index]:.4f}"
        )

        print(
            row["passage"][:250]
        )


# =========================================================
# RUN BOTH
# =========================================================

print("\n")
print("##########################################")
print("ASSAMESE QUERY TEST")
print("##########################################")

test_query(
    ASSAMESE_QUERY
)


print("\n")
print("##########################################")
print("ENGLISH QUERY TEST")
print("##########################################")

test_query(
    ENGLISH_QUERY
)


print("\n")
print("==========================================")
print("COMPARISON COMPLETED")
print("==========================================")