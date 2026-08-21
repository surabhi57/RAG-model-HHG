import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

MODEL_NAME = "sentence-transformers/LaBSE"

ASSAMESE_QUERY = "কৰ্পোৰেচন কি?"
ENGLISH_QUERY = "what is a corporation?"

CORRECT_PASSAGE_ID = 5

print("=" * 60)
print("QUERY TRANSLATION RETRIEVAL EXPERIMENT")
print("=" * 60)

print("\nReading CSV...")

df = pd.read_csv(CSV_FILE, encoding="utf-8")

print("Rows:", len(df))

df = df.drop_duplicates(subset=["passage_id"]).reset_index(drop=True)

print("Unique passages:", len(df))

print("\nLoading model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")

print("\nCreating passage embeddings...")

passages = df["passage"].fillna("").astype(str).tolist()

passage_embeddings = model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Passage embeddings created.")


def retrieve(query, top_k=10):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        passage_embeddings
    )[0]

    rankings = np.argsort(similarities)[::-1]

    results = []

    for rank, index in enumerate(rankings[:top_k], start=1):

        results.append({
            "rank": rank,
            "passage_id": df.iloc[index]["passage_id"],
            "similarity": similarities[index],
            "passage": df.iloc[index]["passage"]
        })

    return results


print("\n")
print("=" * 60)
print("TEST 1: ORIGINAL ASSAMESE QUERY")
print("=" * 60)

print("Query:", ASSAMESE_QUERY)

results_assamese = retrieve(ASSAMESE_QUERY, top_k=10)

correct_rank_assamese = None

for result in results_assamese:

    print("\nRank:", result["rank"])
    print("Passage ID:", result["passage_id"])
    print("Similarity:", round(result["similarity"], 4))
    print("Passage:", result["passage"][:300])

    if int(result["passage_id"]) == CORRECT_PASSAGE_ID:
        correct_rank_assamese = result["rank"]


print("\n")
print("=" * 60)
print("TEST 2: ENGLISH TRANSLATED QUERY")
print("=" * 60)

print("Original Assamese:", ASSAMESE_QUERY)
print("English query:", ENGLISH_QUERY)

results_english = retrieve(ENGLISH_QUERY, top_k=10)

correct_rank_english = None

for result in results_english:

    print("\nRank:", result["rank"])
    print("Passage ID:", result["passage_id"])
    print("Similarity:", round(result["similarity"], 4))
    print("Passage:", result["passage"][:300])

    if int(result["passage_id"]) == CORRECT_PASSAGE_ID:
        correct_rank_english = result["rank"]


print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print("\nOriginal Assamese query:")
print("Correct passage rank:", correct_rank_assamese)

print("\nEnglish translated query:")
print("Correct passage rank:", correct_rank_english)

print("\n")

if correct_rank_assamese is not None:
    if correct_rank_assamese <= 5:
        print("Assamese Recall@5: 100%")
    else:
        print("Assamese Recall@5: 0%")
else:
    print("Assamese Recall@5: 0%")


if correct_rank_english is not None:
    if correct_rank_english <= 5:
        print("English Recall@5: 100%")
    else:
        print("English Recall@5: 0%")
else:
    print("English Recall@5: 0%")


print("\n")
print("=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)