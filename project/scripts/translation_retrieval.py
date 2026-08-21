import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts"

ASSAMESE_CSV = os.path.join(BASE_DIR, "passages.csv")
ENGLISH_CSV = os.path.join(BASE_DIR, "english_passages.csv")

EMBEDDING_MODEL = "sentence-transformers/LaBSE"

TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

TOP_K = 5


# ============================================================
# START
# ============================================================

print("\n")
print("=" * 70)
print("TRANSLATED RETRIEVAL SYSTEM")
print("=" * 70)

print("\nAssamese CSV:")
print(ASSAMESE_CSV)

print("\nEnglish CSV:")
print(ENGLISH_CSV)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(ASSAMESE_CSV):
    print("\nERROR: passages.csv not found.")
    print(ASSAMESE_CSV)
    input("\nPress Enter to exit...")
    raise SystemExit

if not os.path.exists(ENGLISH_CSV):
    print("\nERROR: english_passages.csv not found.")
    print(ENGLISH_CSV)
    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# LOAD ENGLISH PASSAGES
# ============================================================

print("\n")
print("=" * 70)
print("LOADING ENGLISH PASSAGES")
print("=" * 70)

df = pd.read_csv(ENGLISH_CSV)

print("\nEnglish CSV loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "passage" not in df.columns:
    print("\nERROR: 'passage' column is missing.")
    print("Available columns:", list(df.columns))
    input("\nPress Enter to exit...")
    raise SystemExit


# Remove empty passages

df = df.dropna(subset=["passage"]).reset_index(drop=True)

print("Valid English passages:", len(df))


# ============================================================
# LOAD TRANSLATION MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING TRANSLATION MODEL")
print("=" * 70)

print("\nModel:", TRANSLATION_MODEL)

tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL
)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

translation_model = translation_model.to(device)

print("\nTranslation device:", device)

print("Translation model loaded successfully.")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING ENGLISH EMBEDDING MODEL")
print("=" * 70)

print("\nModel:", EMBEDDING_MODEL)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("\nEmbedding model loaded successfully.")


# ============================================================
# CREATE ENGLISH PASSAGE EMBEDDINGS
# ============================================================

print("\n")
print("=" * 70)
print("CREATING ENGLISH PASSAGE EMBEDDINGS")
print("=" * 70)

english_passages = df["passage"].astype(str).tolist()

english_embeddings = embedding_model.encode(
    english_passages,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\nEnglish embeddings created.")

print(
    "Embedding shape:",
    english_embeddings.shape
)


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_assamese_to_english(text):

    # Assamese language code in NLLB
    source_language = "asm_Beng"

    # English language code
    target_language = "eng_Latn"

    tokenizer.src_lang = source_language

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        target_language
    )

    with torch.no_grad():

        generated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )

    translated_text = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_passages(query, top_k=5):

    print("\n")
    print("=" * 70)
    print("TRANSLATING QUERY")
    print("=" * 70)

    print("\nAssamese Question:")
    print(query)

    english_query = translate_assamese_to_english(
        query
    )

    print("\nEnglish translated query:")
    print(english_query)

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [english_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarities = (
        english_embeddings @ query_embedding
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP", top_k, "RETRIEVED ENGLISH PASSAGES")
    print("=" * 70)

    for rank, index in enumerate(
        ranked_indices[:top_k],
        start=1
    ):

        print("\nRank:", rank)

        if "passage_id" in df.columns:
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
        print(df.iloc[index]["passage"])

        print("\n" + "-" * 70)

    return english_query, ranked_indices


# ============================================================
# QUERY LOOP
# ============================================================

print("\n")
print("=" * 70)
print("TRANSLATED RETRIEVAL SYSTEM READY")
print("=" * 70)

print("\nFlow:")
print("Assam")