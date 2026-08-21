import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\english_passages.csv"

MODEL_NAME = "sentence-transformers/LaBSE"

TOP_K = 5


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ENGLISH RETRIEVAL TEST")
print("=" * 70)

print("\nCSV FILE:")
print(CSV_FILE)


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading English passages...")
# Load English passages
import pandas as pd
import os

ENGLISH_CSV = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\english_passages.csv"

print("Loading English passages...")

if not os.path.exists(ENGLISH_CSV):
    print("ERROR: english_passages.csv not found!")
    print("Expected location:")
    print(ENGLISH_CSV)
    exit()

df = pd.read_csv(ENGLISH_CSV)

print("English CSV loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nEnglish CSV loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))

pd.read_csv(CSV_FILE)

print("CSV loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(subset=["passage"])

df["passage"] = df["passage"].astype(str)

print("\nValid passages:", len(df))


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

print("\nModel:", MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# CREATE PASSAGE EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("CREATING ENGLISH PASSAGE EMBEDDINGS")
print("=" * 70)

passages = df["passage"].tolist()

embeddings = model.encode(
    passages,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\nEmbeddings created.")
print("Embedding shape:", embeddings.shape)


# ============================================================
# QUERY
# ============================================================

query = input(
    "\nEnter an English question "
    "(type 'exit' to stop): "
)


if query.lower() == "exit":

    print("Program stopped.")

else:

    print("\nSearching...")

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarities = embeddings @ query_embedding


    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        similarities
    )[::-1]


    # --------------------------------------------------------
    # Display TOP K
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP", TOP_K, "RETRIEVED ENGLISH PASSAGES")
    print("=" * 70)


    for rank, index in enumerate(
        ranked_indices[:TOP_K],
        start=1
    ):

        print("\nRank:", rank)

        print(
            "Passage ID:",
            df.iloc[index]["passage_id"]
        )

        print(
            "Similarity:",
            round(
                float(similarities[index]),
                4
            )
        )

        print("\nPassage:")

        print(
            df.iloc[index]["passage"]
        )

        print("\n" + "-" * 70)


print("\n" + "=" * 70)
print("ENGLISH RETRIEVAL TEST COMPLETED")
print("=" * 70)