import os
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts"

EVALUATION_CSV = os.path.join(
    BASE_DIR,
    "evaluation_queries.csv"
)

ENGLISH_CSV = os.path.join(
    BASE_DIR,
    "english_passages.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "translated_retrieval_evaluation.csv"
)


# Translation model
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

# Assamese -> English
SOURCE_LANGUAGE = "asm_Beng"
TARGET_LANGUAGE = "eng_Latn"

# Retrieval model
EMBEDDING_MODEL = "sentence-transformers/LaBSE"

TOP_K = 10


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MULTILINGUAL TRANSLATED RETRIEVAL EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking files...")

if not os.path.exists(EVALUATION_CSV):
    raise FileNotFoundError(
        f"Evaluation file not found:\n{EVALUATION_CSV}"
    )

if not os.path.exists(ENGLISH_CSV):
    raise FileNotFoundError(
        f"English passages file not found:\n{ENGLISH_CSV}"
    )

print("Evaluation file found.")
print("English passages file found.")


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING EVALUATION DATA")
print("=" * 70)

eval_df = pd.read_csv(
    EVALUATION_CSV,
    encoding="utf-8-sig"
)

print("\nEvaluation queries:", len(eval_df))

print(
    "Columns:",
    eval_df.columns.tolist()
)


required_columns = [
    "query_id",
    "query",
    "expected_passage_id"
]

for column in required_columns:

    if column not in eval_df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# LOAD ENGLISH PASSAGES
# ============================================================

print("\n" + "=" * 70)
print("LOADING ENGLISH PASSAGES")
print("=" * 70)

passage_df = pd.read_csv(
    ENGLISH_CSV,
    encoding="utf-8-sig"
)

passage_df["passage"] = (
    passage_df["passage"]
    .fillna("")
    .astype(str)
)

print(
    "\nEnglish passages:",
    len(passage_df)
)

print(
    "Columns:",
    passage_df.columns.tolist()
)


if "passage_id" not in passage_df.columns:

    raise ValueError(
        "english_passages.csv must contain passage_id"
    )


# ============================================================
# LOAD TRANSLATION MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRANSLATION MODEL")
print("=" * 70)

print(
    "\nModel:",
    TRANSLATION_MODEL
)

print(
    "\nThis may take some time the first time."
)

translation_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL
)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL
)


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_assamese_to_english(text):

    text = str(text).strip()

    if not text:

        return ""

    # Tell NLLB that the input language is Assamese
    translation_tokenizer.src_lang = SOURCE_LANGUAGE

    inputs = translation_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():

        generated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=
            translation_tokenizer.convert_tokens_to_ids(
                TARGET_LANGUAGE
            ),
            max_new_tokens=128
        )

    translated_text = (
        translation_tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        .strip()
    )

    return translated_text


# ============================================================
# TEST TRANSLATION
# ============================================================

print("\n" + "=" * 70)
print("TESTING TRANSLATION")
print("=" * 70)

test_query = str(
    eval_df.iloc[0]["query"]
)

print("\nOriginal Assamese:")
print(test_query)

test_translation = translate_assamese_to_english(
    test_query
)

print("\nEnglish translation:")
print(test_translation)


# ============================================================
# LOAD LaBSE
# ============================================================

print("\n" + "=" * 70)
print("LOADING LaBSE")
print("=" * 70)

print(
    "\nModel:",
    EMBEDDING_MODEL
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("\nLaBSE loaded successfully.")


# ============================================================
# CREATE ENGLISH PASSAGE EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("CREATING ENGLISH PASSAGE EMBEDDINGS")
print("=" * 70)

passages = passage_df[
    "passage"
].tolist()

embeddings = embedding_model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(
    "\nEmbedding shape:",
    embeddings.shape
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING MULTILINGUAL RETRIEVAL EVALUATION")
print("=" * 70)


recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0

reciprocal_ranks = []

results = []


total_queries = len(eval_df)


for i, row in eval_df.iterrows():

    query = str(
        row["query"]
    ).strip()

    expected_id = str(
        row["expected_passage_id"]
    ).strip()


    print("\n" + "-" * 70)

    print(
        f"Query {i + 1}/{total_queries}"
    )

    print(
        "Original:",
        query
    )


    # --------------------------------------------------------
    # TRANSLATE
    # --------------------------------------------------------

    english_query = translate_assamese_to_english(
        query
    )

    print(
        "English:",
        english_query
    )


    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
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
    # FIND CORRECT PASSAGE
    # --------------------------------------------------------

    retrieved_ids = []

    correct_rank = None


    for rank, index in enumerate(
        ranked_indices[:TOP_K],
        start=1
    ):

        retrieved_id = str(
            passage_df.iloc[index]["passage_id"]
        ).strip()

        retrieved_ids.append(
            retrieved_id
        )


        if (
            retrieved_id == expected_id
            and correct_rank is None
        ):

            correct_rank = rank


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if correct_rank == 1:

        recall_at_1 += 1


    if (
        correct_rank is not None
        and correct_rank <= 5
    ):

        recall_at_5 += 1


    if (
        correct_rank is not None
        and correct_rank <= 10
    ):

        recall_at_10 += 1


    if correct_rank is not None:

        reciprocal_ranks.append(
            1.0 / correct_rank
        )

    else:

        reciprocal_ranks.append(
            0.0
        )


    print(
        "Expected passage:",
        expected_id
    )

    print(
        "Correct rank:",
        correct_rank
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "query_id":
                row["query_id"],

            "query":
                query,

            "english_query":
                english_query,

            "expected_passage_id":
                expected_id,

            "correct_rank":
                correct_rank,

            "top_10_ids":
                ",".join(
                    retrieved_ids
                )
        }
    )


# ============================================================
# FINAL METRICS
# ============================================================

total = len(eval_df)


recall_at_1_score = (
    recall_at_1 / total
)

recall_at_5_score = (
    recall_at_5 / total
)

recall_at_10_score = (
    recall_at_10 / total
)

mrr = np.mean(
    reciprocal_ranks
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL MULTILINGUAL RETRIEVAL METRICS")
print("=" * 70)

print(
    "\nTotal queries:",
    total
)

print(
    "\nRecall@1:",
    round(
        recall_at_1_score * 100,
        2
    ),
    "%"
)

print(
    "Recall@5:",
    round(
        recall_at_5_score * 100,
        2
    ),
    "%"
)

print(
    "Recall@10:",
    round(
        recall_at_10_score * 100,
        2
    ),
    "%"
)

print(
    "MRR:",
    round(
        mrr,
        4
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nDetailed results saved to:"
)

print(
    OUTPUT_CSV
)


print("\n" + "=" * 70)
print("MULTILINGUAL EVALUATION COMPLETED")
print("=" * 70)