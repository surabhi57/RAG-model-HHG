import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(
    BASE_DIR,
    "passages.csv"
)

MODEL_NAME = "sentence-transformers/LaBSE"

TOP_K = 5


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("STARTING RAG SYSTEM")
print("=" * 70)

print()
print("CSV FILE:")
print(CSV_FILE)


# ============================================================
# CHECK CSV
# ============================================================

if not os.path.exists(CSV_FILE):

    print()
    print("ERROR: passages.csv was not found.")
    print("Expected location:")
    print(CSV_FILE)

    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading passages.csv...")

try:

    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8"
    )

except UnicodeDecodeError:

    print("UTF-8 failed. Trying UTF-8-SIG...")

    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8-sig"
    )


print()
print("CSV loaded successfully.")

print("Rows:", len(df))

print(
    "Columns:",
    list(df.columns)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "passage_id",
    "query",
    "passage"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    print()
    print("=" * 70)
    print("ERROR: REQUIRED COLUMNS ARE MISSING")
    print("=" * 70)

    print(
        "Missing:",
        missing_columns
    )

    print()
    print("Available columns:")
    print(list(df.columns))

    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# CLEAN DATA
# ============================================================

df = df.fillna("")


df["passage_id"] = df[
    "passage_id"
].astype(str)


df["passage"] = df[
    "passage"
].astype(str)


# Remove completely empty passages

df = df[
    df["passage"].str.strip() != ""
].reset_index(drop=True)


print()
print("Valid passages:", len(df))


# ============================================================
# CREATE PASSAGE LIST
# ============================================================

passages = []

for index, row in df.iterrows():

    passages.append(
        {
            "index": index,
            "passage_id": row["passage_id"],
            "text": row["passage"]
        }
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

print()
print("Model:", MODEL_NAME)

try:

    model = SentenceTransformer(
        MODEL_NAME
    )

except Exception as e:

    print()
    print("ERROR WHILE LOADING MODEL:")
    print(e)

    input("\nPress Enter to exit...")
    raise SystemExit


print()
print("Model loaded successfully.")


# ============================================================
# CREATE PASSAGE EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("CREATING PASSAGE EMBEDDINGS")
print("=" * 70)

passage_texts = [
    item["text"]
    for item in passages
]


try:

    embeddings = model.encode(
        passage_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

except Exception as e:

    print()
    print("ERROR WHILE CREATING EMBEDDINGS:")
    print(e)

    input("\nPress Enter to exit...")
    raise SystemExit


print()
print("Embeddings created.")

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# RAG SYSTEM READY
# ============================================================

print()
print("=" * 70)
print("RAG SYSTEM READY")
print("=" * 70)

print()
print("Model:", MODEL_NAME)
print("Passages:", len(passages))
print("Top-K:", TOP_K)

print()
print("You can now enter your question.")
print("Type 'exit' to stop.")


# ============================================================
# QUERY LOOP
# ============================================================

while True:

    print()

    query = input(
        "Enter your question: "
    ).strip()


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if query.lower() == "exit":

        print()
        print("=" * 70)
        print("RAG SYSTEM STOPPED")
        print("=" * 70)

        break


    # --------------------------------------------------------
    # EMPTY QUERY
    # --------------------------------------------------------

    if not query:

        print(
            "Please enter a question."
        )

        continue


    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    print()
    print("Searching...")


    try:

        query_embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    except Exception as e:

        print()
        print("ERROR CREATING QUERY EMBEDDING:")
        print(e)

        continue


    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    similarities = (
        embeddings @ query_embedding
    )


    # --------------------------------------------------------
    # RANK RESULTS
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        similarities
    )[::-1]


    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    top_indices = ranked_indices[
        :TOP_K
    ]


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP", TOP_K, "RETRIEVED PASSAGES")
    print("=" * 70)


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        passage = passages[index]

        similarity = float(
            similarities[index]
        )


        print()
        print("Rank:", rank)

        print(
            "Passage ID:",
            passage["passage_id"]
        )

        print(
            "Similarity:",
            round(
                similarity,
                4
            )
        )

        print()
        print("Passage:")

        print(
            passage["text"]
        )

        print()
        print("-" * 70)


    # --------------------------------------------------------
    # CONTEXT FOR NEXT LLM STAGE
    # --------------------------------------------------------

    retrieved_context = "\n\n".join(
        passages[index]["text"]
        for index in top_indices
    )


    print()
    print("=" * 70)
    print("RETRIEVED CONTEXT FOR GENERATION")
    print("=" * 70)

    print()
    print(retrieved_context)

    print()
    print("=" * 70)
    print("RETRIEVAL COMPLETED")
    print("=" * 70)