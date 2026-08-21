import os
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


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

# Multilingual translation model
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MULTILINGUAL RAG SYSTEM")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

print("\nChecking English dataset...")

if not os.path.exists(ENGLISH_CSV):

    print("\nERROR: English dataset not found.")

    print(
        "Expected:"
    )

    print(ENGLISH_CSV)

    raise SystemExit


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading English passages...")

df = pd.read_csv(
    ENGLISH_CSV,
    encoding="utf-8-sig"
)

df["passage"] = (
    df["passage"]
    .fillna("")
    .astype(str)
)

passages = df["passage"].tolist()

print("\nDataset loaded.")

print("Rows:", len(df))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

print("\nModel:")
print(EMBEDDING_MODEL)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("\nEmbedding model loaded.")


# ============================================================
# CREATE PASSAGE EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("CREATING PASSAGE EMBEDDINGS")
print("=" * 70)

passage_embeddings = embedding_model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("\nEmbeddings created.")

print(
    "Embedding shape:",
    passage_embeddings.shape
)


# ============================================================
# LOAD TRANSLATION MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MULTILINGUAL TRANSLATION MODEL")
print("=" * 70)

print("\nModel:")
print(TRANSLATION_MODEL)

tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL
)

translator = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL
)

print("\nTranslation model loaded.")


# ============================================================
# LANGUAGE CODES
# ============================================================

LANGUAGE_CODES = {

    "English": "eng_Latn",

    "Assamese": "asm_Beng",

    "Hindi": "hin_Deva",

    "Kannada": "kan_Knda",

    "Bengali": "ben_Beng",

    "Tamil": "tam_Taml",

    "Telugu": "tel_Telu",

    "Malayalam": "mal_Mlym",

    "Marathi": "mar_Deva",

    "Gujarati": "guj_Gujr",

    "Punjabi": "pan_Guru",

    "Urdu": "urd_Arab"
}


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_text(
    text,
    source_language,
    target_language
):

    source_code = LANGUAGE_CODES[
        source_language
    ]

    target_code = LANGUAGE_CODES[
        target_language
    ]

    tokenizer.src_lang = source_code

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    forced_bos_token_id = (
        tokenizer.convert_tokens_to_ids(
            target_code
        )
    )

    generated_tokens = translator.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=100
    )

    translated_text = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


# ============================================================
# READY
# ============================================================

print("\n" + "=" * 70)
print("MULTILINGUAL RETRIEVAL READY")
print("=" * 70)

print("\nSupported languages:")

for language in LANGUAGE_CODES:
    print("-", language)

print("\nPipeline:")

print(
    "User language"
    " → English"
    " → Retrieval"
)

print("\nType 'exit' to stop.")


# ============================================================
# QUERY LOOP
# ============================================================

while True:

    print("\n" + "=" * 70)

    language = input(
        "Enter language name: "
    ).strip()

    if language.lower() == "exit":
        break

    if language not in LANGUAGE_CODES:

        print(
            "\nUnsupported language."
        )

        print(
            "Use one of:"
        )

        print(
            ", ".join(LANGUAGE_CODES.keys())
        )

        continue


    query = input(
        "Enter your question: "
    ).strip()


    if query.lower() == "exit":
        break


    # ========================================================
    # TRANSLATE TO ENGLISH
    # ========================================================

    if language == "English":

        english_query = query

    else:

        print(
            "\nTranslating question to English..."
        )

        english_query = translate_text(
            query,
            language,
            "English"
        )


    print("\nOriginal question:")

    print(query)

    print("\nEnglish query:")

    print(english_query)


    # ========================================================
    # CREATE QUERY EMBEDDING
    # ========================================================

    query_embedding = embedding_model.encode(
        [english_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]


    # ========================================================
    # SIMILARITY
    # ========================================================

    similarities = (
        passage_embeddings @ query_embedding
    )


    # ========================================================
    # RANK
    # ========================================================

    ranked_indices = np.argsort(
        similarities
    )[::-1]


    # ========================================================
    # DISPLAY TOP 5
    # ========================================================

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

        print(
            "\nRank:",
            rank
        )

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

        print(
            "\n" + "-" * 70
        )


print("\nMultilingual retrieval stopped.")