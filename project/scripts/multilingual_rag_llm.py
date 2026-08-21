import os
import torch
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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

TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

LLM_MODEL = "google/flan-t5-small"


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
# START
# ============================================================

print("=" * 70)
print("MULTILINGUAL RAG + LLM SYSTEM")
print("=" * 70)


# ============================================================
# CHECK CSV
# ============================================================

print("\nChecking English dataset...")

if not os.path.exists(ENGLISH_CSV):

    print("\nERROR: english_passages.csv not found.")

    print("Expected location:")
    print(ENGLISH_CSV)

    raise SystemExit


# ============================================================
# LOAD CSV
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

print("Dataset loaded.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

print("\nModel:", EMBEDDING_MODEL)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("\nEmbedding model loaded.")


# ============================================================
# CREATE EMBEDDINGS
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

print("\nModel:", TRANSLATION_MODEL)

translation_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL
)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL
)

print("\nTranslation model loaded.")


# ============================================================
# LOAD LLM
# ============================================================

print("\n" + "=" * 70)
print("LOADING LLM")
print("=" * 70)

print("\nModel:", LLM_MODEL)

llm_tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)

print("\nLLM loaded successfully.")


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

    translation_tokenizer.src_lang = source_code

    inputs = translation_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    target_token_id = (
        translation_tokenizer.convert_tokens_to_ids(
            target_code
        )
    )

    with torch.no_grad():

        outputs = translation_model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_new_tokens=100
        )

    translated = translation_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated.strip()


# ============================================================
# LLM FUNCTION
# ============================================================

def generate_answer(
    question,
    context
):

    prompt = (
        "Answer the question using ONLY the information "
        "provided in the context.\n\n"
        "If the answer is not present in the context, "
        "say that the information is not available.\n\n"
        "Context:\n"
        + context
        + "\n\n"
        "Question:\n"
        + question
        + "\n\n"
        "Answer:"
    )

    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True
        )

    answer = llm_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# READY
# ============================================================

print("\n" + "=" * 70)
print("MULTILINGUAL RAG SYSTEM READY")
print("=" * 70)

print("\nPipeline:")

print(
    "Multiple languages"
    " → English translation"
    " → English retrieval"
    " → LLM"
    " → Answer in original language"
)

print("\nSupported languages:")

for language in LANGUAGE_CODES:
    print("-", language)

print("\nTop-K:", TOP_K)

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

        print("\nRAG system stopped.")
        break

    if language not in LANGUAGE_CODES:

        print("\nUnsupported language.")

        print(
            "\nSupported languages:"
        )

        print(
            ", ".join(
                LANGUAGE_CODES.keys()
            )
        )

        continue

    query = input(
        "Enter your question: "
    ).strip()

    if query.lower() == "exit":

        print("\nRAG system stopped.")
        break


    # ========================================================
    # TRANSLATE QUERY
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 1: TRANSLATING QUESTION TO ENGLISH")
    print("=" * 70)

    if language == "English":

        english_query = query

    else:

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
    # RETRIEVAL
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 2: ENGLISH RETRIEVAL")
    print("=" * 70)

    query_embedding = embedding_model.encode(
        [english_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    similarities = (
        passage_embeddings @ query_embedding
    )

    ranked_indices = np.argsort(
        similarities
    )[::-1]


    print("\n" + "=" * 70)
    print("TOP", TOP_K, "RETRIEVED ENGLISH PASSAGES")
    print("=" * 70)

    retrieved_context = []

    for rank, index in enumerate(
        ranked_indices[:TOP_K],
        start=1
    ):

        passage_text = df.iloc[index]["passage"]

        retrieved_context.append(
            passage_text
        )

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
        print(passage_text)

        print("\n" + "-" * 70)


    # ========================================================
    # GENERATE ENGLISH ANSWER
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 3: GENERATING ENGLISH ANSWER")
    print("=" * 70)

    context = "\n\n".join(
        retrieved_context
    )

    english_answer = generate_answer(
        english_query,
        context
    )

    print("\nEnglish answer:")
    print(english_answer)


    # ========================================================
    # TRANSLATE ANSWER BACK
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 4: TRANSLATING ANSWER BACK")
    print("=" * 70)

    if language == "English":

        final_answer = english_answer

    else:

        final_answer = translate_text(
            english_answer,
            "English",
            language
        )


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print("\nQuestion:")
    print(query)

    print("\nAnswer:")
    print(final_answer)

    print("\n" + "=" * 70)