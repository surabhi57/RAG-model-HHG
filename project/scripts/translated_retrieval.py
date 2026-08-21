import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import MarianTokenizer, MarianMTModel

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts"

ENGLISH_CSV = os.path.join(
    BASE_DIR,
    "english_passages.csv"
)

TOP_K = 5

EMBEDDING_MODEL = "sentence-transformers/LaBSE"

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-inc-en"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TRANSLATED RETRIEVAL SYSTEM")
print("=" * 70)

print("\nEnglish CSV:")
print(ENGLISH_CSV)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(ENGLISH_CSV):

    print("\nERROR:")
    print("english_passages.csv was not found.")

    print("\nExpected location:")
    print(ENGLISH_CSV)

    raise SystemExit


# ============================================================
# LOAD ENGLISH DATASET
# ============================================================

print("\nLoading English passages...")

df = pd.read_csv(
    ENGLISH_CSV,
    encoding="utf-8-sig"
)

print("English dataset loaded.")

print("Rows:", len(df))

print(
    "Columns:",
    list(df.columns)
)


# ============================================================
# CLEAN DATA
# ============================================================

df["passage"] = (
    df["passage"]
    .fillna("")
    .astype(str)
)

df["query"] = (
    df["query"]
    .fillna("")
    .astype(str)
)


passages = df["passage"].tolist()


print("\nValid passages:", len(passages))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

print("\nModel:")
print(EMBEDDING_MODEL)


model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("\nEmbedding model loaded.")


# ============================================================
# CREATE PASSAGE EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("CREATING ENGLISH PASSAGE EMBEDDINGS")
print("=" * 70)


embeddings = model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


print("\nEmbeddings created.")

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# LOAD TRANSLATION MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING ASSAMESE → ENGLISH TRANSLATION MODEL")
print("=" * 70)

print("\nModel:")
print(TRANSLATION_MODEL)


tokenizer = MarianTokenizer.from_pretrained(
    TRANSLATION_MODEL
)

translator = MarianMTModel.from_pretrained(
    TRANSLATION_MODEL
)

print("\nTranslation model loaded.")


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_assamese_to_english(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    )

    outputs = translator.generate(
        **inputs,
        max_new_tokens=100
    )

    translated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated


# ============================================================
# RAG SYSTEM READY
# ============================================================

print("\n" + "=" * 70)
print("TRANSLATED RAG SYSTEM READY")
print("=" * 70)

print("\nEmbedding model:")
print(EMBEDDING_MODEL)

print("Translation model:")
print(TRANSLATION_MODEL)

print("English passages:", len(passages))

print("Top-K:", TOP_K)

print("\nPipeline:")

print(
    "Assamese Question"
    " → English Translation"
    " → English Retrieval"
)

print("\nYou can now enter your Assamese question.")

print("Type 'exit' to stop.")


# ============================================================
# QUERY LOOP
# ============================================================

while True:

    print("\n" + "=" * 70)

    query = input(
        "Enter your Assamese question: "
    )

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if query.strip().lower() == "exit":

        print("\nRAG system stopped.")

        break


    # --------------------------------------------------------
    # TRANSLATE
    # --------------------------------------------------------

    print("\nTranslating query...")

    english_query = translate_assamese_to_english(
        query
    )


    print("\nOriginal Assamese query:")

    print(query)


    print("\nEnglish translated query:")

    print(english_query)


    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    print("\nCreating query embedding...")

    query_embedding = model.encode(
        [english_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]


    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    similarities = (
        embeddings @ query_embedding
    )


    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        similarities
    )[::-1]


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "TOP",
        TOP_K,
        "RETRIEVED ENGLISH PASSAGES"
    )

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
                float(
                    similarities[index]
                ),
                4
            )
        )

        print("\nPassage:")

        print(
            df.iloc[index]["passage"]
        )

        print("\n" + "-" * 70)


    # --------------------------------------------------------
    # RETRIEVED CONTEXT
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("RETRIEVED ENGLISH CONTEXT")

    print("=" * 70)


    for index in ranked_indices[:TOP_K]:

        print(
            df.iloc[index]["passage"]
        )

        print()


    print("=" * 70)

    print("RETRIEVAL COMPLETED")

    print("=" * 70)