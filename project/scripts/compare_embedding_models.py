import csv
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

EXPECTED_QUERY_ID = "1102432"
EXPECTED_PASSAGE_ID = "5"

ASSAMESE_QUERY = "কৰ্পোৰেচন কি?"

ENGLISH_QUERY = "what is a corporation?"


MODELS = [

    "sentence-transformers/LaBSE",

    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",

    "sentence-transformers/distiluse-base-multilingual-cased-v2"

]


# =========================================================
# LOAD DATA
# =========================================================

print("==========================================")
print("EMBEDDING MODEL COMPARISON")
print("==========================================")


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


print(
    "Unique passages:",
    len(corpus)
)


texts = [
    row["passage"]
    for row in corpus
]


# =========================================================
# EXPECTED PASSAGE
# =========================================================

expected_index = None


for i, row in enumerate(corpus):

    if (
        row["query_id"] == EXPECTED_QUERY_ID
        and row["passage_id"] == EXPECTED_PASSAGE_ID
    ):

        expected_index = i

        break


if expected_index is None:

    print(
        "ERROR: Expected passage not found."
    )

    raise SystemExit


# =========================================================
# TEST FUNCTION
# =========================================================

def evaluate_query(
    model,
    embeddings,
    query
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]


    query_embedding = (
        query_embedding /
        np.linalg.norm(query_embedding)
    )


    similarities = (
        embeddings @
        query_embedding
    )


    ranking = np.argsort(
        similarities
    )[::-1]


    rank = (
        np.where(
            ranking == expected_index
        )[0][0] + 1
    )


    if rank <= 1:

        recall_1 = 1

    else:

        recall_1 = 0


    if rank <= 3:

        recall_3 = 1

    else:

        recall_3 = 0


    if rank <= 5:

        recall_5 = 1

    else:

        recall_5 = 0


    reciprocal_rank = 1 / rank


    return (
        rank,
        similarities[expected_index],
        recall_1,
        recall_3,
        recall_5,
        reciprocal_rank
    )


# =========================================================
# RUN MODELS
# =========================================================

for model_name in MODELS:

    print("\n\n")
    print("==========================================")
    print("MODEL")
    print(model_name)
    print("==========================================")


    print("\nLoading model...")


    model = SentenceTransformer(
        model_name
    )


    print("Model loaded.")


    # -----------------------------------------------------
    # PASSAGE EMBEDDINGS
    # -----------------------------------------------------

    print(
        "\nCreating passage embeddings..."
    )


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


    # =====================================================
    # ASSAMESE
    # =====================================================

    result_assamese = evaluate_query(
        model,
        embeddings,
        ASSAMESE_QUERY
    )


    print("\nASSAMESE QUERY")

    print(
        "Query:",
        ASSAMESE_QUERY
    )

    print(
        "Correct rank:",
        result_assamese[0]
    )

    print(
        "Similarity:",
        f"{result_assamese[1]:.4f}"
    )

    print(
        "Recall@1:",
        result_assamese[2]
    )

    print(
        "Recall@3:",
        result_assamese[3]
    )

    print(
        "Recall@5:",
        result_assamese[4]
    )

    print(
        "MRR:",
        f"{result_assamese[5]:.4f}"
    )


    # =====================================================
    # ENGLISH
    # =====================================================

    result_english = evaluate_query(
        model,
        embeddings,
        ENGLISH_QUERY
    )


    print("\nENGLISH QUERY")

    print(
        "Query:",
        ENGLISH_QUERY
    )

    print(
        "Correct rank:",
        result_english[0]
    )

    print(
        "Similarity:",
        f"{result_english[1]:.4f}"
    )

    print(
        "Recall@1:",
        result_english[2]
    )

    print(
        "Recall@3:",
        result_english[3]
    )

    print(
        "Recall@5:",
        result_english[4]
    )

    print(
        "MRR:",
        f"{result_english[5]:.4f}"
    )


print("\n")
print("==========================================")
print("MODEL COMPARISON COMPLETED")
print("==========================================")