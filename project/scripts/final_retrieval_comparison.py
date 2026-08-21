import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

MODEL_NAME = "sentence-transformers/LaBSE"


print("=" * 70)
print("FINAL RETRIEVAL COMPARISON")
print("=" * 70)


# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------

print("\nReading passages.csv...")

df = pd.read_csv(
    CSV_FILE,
    encoding="utf-8"
)

print("Total rows:", len(df))


# Remove duplicate passage IDs
df = df.drop_duplicates(
    subset=["passage_id"]
).reset_index(drop=True)

print("Unique passages:", len(df))


# ---------------------------------------------------------
# STEP 2: LOAD MODEL
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ---------------------------------------------------------
# STEP 3: CREATE PASSAGE EMBEDDINGS
# ---------------------------------------------------------

print("\nCreating passage embeddings...")

passages = (
    df["passage"]
    .fillna("")
    .astype(str)
    .tolist()
)

passage_embeddings = model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embeddings created.")


# ---------------------------------------------------------
# STEP 4: RETRIEVAL FUNCTION
# ---------------------------------------------------------

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

    ranking = np.argsort(
        similarities
    )[::-1]

    results = []

    for rank, index in enumerate(
        ranking[:top_k],
        start=1
    ):

        results.append({
            "rank": rank,
            "passage_id": df.iloc[index]["passage_id"],
            "similarity": similarities[index],
            "passage": df.iloc[index]["passage"]
        })

    return results


# ---------------------------------------------------------
# STEP 5: TEST QUERIES
# ---------------------------------------------------------

queries = [

    {
        "name": "Corporation",
        "assamese": "কৰ্পোৰেচন কি?",
        "english": "what is a corporation?",
        "correct_id": 5
    },

]


# ---------------------------------------------------------
# STEP 6: RUN EXPERIMENT
# ---------------------------------------------------------

all_results = []


for item in queries:

    print("\n")
    print("=" * 70)
    print("QUERY:", item["name"])
    print("=" * 70)


    # -----------------------------------------------------
    # ASSAMESE
    # -----------------------------------------------------

    print("\nASSAMESE QUERY")
    print("-----------------------------")

    print(item["assamese"])

    assamese_results = retrieve(
        item["assamese"],
        top_k=10
    )

    assamese_rank = None
    assamese_similarity = None

    for result in assamese_results:

        print(
            f"Rank {result['rank']} | "
            f"ID {result['passage_id']} | "
            f"Similarity {result['similarity']:.4f}"
        )

        if int(result["passage_id"]) == item["correct_id"]:

            assamese_rank = result["rank"]
            assamese_similarity = result["similarity"]


    # -----------------------------------------------------
    # ENGLISH
    # -----------------------------------------------------

    print("\nENGLISH TRANSLATED QUERY")
    print("-----------------------------")

    print(item["english"])

    english_results = retrieve(
        item["english"],
        top_k=10
    )

    english_rank = None
    english_similarity = None

    for result in english_results:

        print(
            f"Rank {result['rank']} | "
            f"ID {result['passage_id']} | "
            f"Similarity {result['similarity']:.4f}"
        )

        if int(result["passage_id"]) == item["correct_id"]:

            english_rank = result["rank"]
            english_similarity = result["similarity"]


    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print("\nRESULT")

    print(
        "Assamese correct rank:",
        assamese_rank
    )

    print(
        "English correct rank:",
        english_rank
    )

    if assamese_rank is not None:
        if assamese_rank <= 5:
            print("Assamese Recall@5: CORRECT")
        else:
            print("Assamese Recall@5: WRONG")
    else:
        print("Assamese Recall@5: WRONG")


    if english_rank is not None:
        if english_rank <= 5:
            print("English Recall@5: CORRECT")
        else:
            print("English Recall@5: WRONG")
    else:
        print("English Recall@5: WRONG")


    # -----------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------

    all_results.append({

        "query": item["assamese"],

        "english_query": item["english"],

        "assamese_rank": assamese_rank,

        "english_rank": english_rank,

        "assamese_similarity": assamese_similarity,

        "english_similarity": english_similarity

    })


# ---------------------------------------------------------
# STEP 7: SAVE CSV
# ---------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

output_file = (
    r"C:\Users\rrutu\OneDrive\Desktop\project"
    r"\scripts\final_retrieval_results.csv"
)

results_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


print("\n")
print("=" * 70)
print("FINAL RESULTS SAVED")
print("=" * 70)

print(output_file)

print("\n")
print(results_df)