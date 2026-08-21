import csv
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"

MODEL_NAME = "sentence-transformers/LaBSE"

CHUNK_SIZES = [50, 100, 200]

OVERLAPS = [0, 10, 20]


# =========================================================
# CHUNKING FUNCTION
# =========================================================

def chunk_text(text, chunk_size, overlap):

    words = text.split()

    chunks = []

    if overlap >= chunk_size:

        raise ValueError(
            "Overlap must be smaller than chunk size."
        )


    start = 0


    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)


        start = end - overlap


    return chunks


# =========================================================
# LOAD DATA
# =========================================================

print("==========================================")
print("LaBSE CHUNKING EXPERIMENT")
print("==========================================")


print("\nReading dataset...")


with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


print(
    "Total dataset rows:",
    len(rows)
)


# =========================================================
# FIND RELEVANT QUERIES
# =========================================================

query_ids = []


for row in rows:

    if row["is_selected"] == "1":

        if row["query_id"] not in query_ids:

            query_ids.append(
                row["query_id"]
            )


print(
    "Relevant unique queries:",
    len(query_ids)
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading LaBSE...")


model = SentenceTransformer(
    MODEL_NAME
)


print("LaBSE loaded successfully.")


# =========================================================
# RESULTS STORAGE
# =========================================================

all_results = []


# =========================================================
# RUN EXPERIMENTS
# =========================================================

for chunk_size in CHUNK_SIZES:

    for overlap in OVERLAPS:

        print("\n")
        print("==========================================")
        print(
            f"CHUNK SIZE: {chunk_size} | "
            f"OVERLAP: {overlap}"
        )
        print("==========================================")


        total_chunks = 0

        recall_1 = 0

        recall_3 = 0

        recall_5 = 0

        reciprocal_ranks = []


        # -------------------------------------------------
        # EACH QUERY
        # -------------------------------------------------

        for query_id in query_ids:

            query_rows = [
                row for row in rows
                if row["query_id"] == query_id
            ]


            if not query_rows:

                continue


            query = query_rows[0]["query"]


            # -------------------------------------------------
            # CORRECT PASSAGE
            # -------------------------------------------------

            correct_passage_ids = {
                row["passage_id"]
                for row in query_rows
                if row["is_selected"] == "1"
            }


            # -------------------------------------------------
            # CREATE CHUNKS
            # -------------------------------------------------

            chunks = []


            for row in query_rows:

                passage_id = row["passage_id"]

                passage = row["passage"]


                passage_chunks = chunk_text(
                    passage,
                    chunk_size,
                    overlap
                )


                for chunk_number, chunk in enumerate(
                    passage_chunks
                ):

                    chunks.append(
                        {
                            "passage_id": passage_id,
                            "chunk_number": chunk_number,
                            "text": chunk
                        }
                    )


            total_chunks += len(chunks)


            # -------------------------------------------------
            # EMBEDDINGS
            # -------------------------------------------------

            chunk_texts = [
                chunk["text"]
                for chunk in chunks
            ]


            chunk_embeddings = model.encode(
                chunk_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )


            chunk_embeddings = (
                chunk_embeddings /
                np.linalg.norm(
                    chunk_embeddings,
                    axis=1,
                    keepdims=True
                )
            )


            query_embedding = model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]


            query_embedding = (
                query_embedding /
                np.linalg.norm(query_embedding)
            )


            # -------------------------------------------------
            # SIMILARITY
            # -------------------------------------------------

            similarities = (
                chunk_embeddings @
                query_embedding
            )


            ranking = np.argsort(
                similarities
            )[::-1]


            # -------------------------------------------------
            # FIND FIRST CORRECT PASSAGE
            # -------------------------------------------------

            first_relevant_rank = None


            for rank, index in enumerate(
                ranking,
                start=1
            ):

                retrieved_passage_id = (
                    chunks[index]["passage_id"]
                )


                if (
                    retrieved_passage_id
                    in correct_passage_ids
                ):

                    first_relevant_rank = rank

                    break


            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------

            if (
                first_relevant_rank is not None
                and first_relevant_rank == 1
            ):

                recall_1 += 1


            if (
                first_relevant_rank is not None
                and first_relevant_rank <= 3
            ):

                recall_3 += 1


            if (
                first_relevant_rank is not None
                and first_relevant_rank <= 5
            ):

                recall_5 += 1


            if first_relevant_rank is not None:

                reciprocal_ranks.append(
                    1 / first_relevant_rank
                )

            else:

                reciprocal_ranks.append(0)


        # -------------------------------------------------
        # CALCULATE METRICS
        # -------------------------------------------------

        total_queries = len(query_ids)


        recall_1_percent = (
            recall_1 /
            total_queries *
            100
        )


        recall_3_percent = (
            recall_3 /
            total_queries *
            100
        )


        recall_5_percent = (
            recall_5 /
            total_queries *
            100
        )


        mrr = np.mean(
            reciprocal_ranks
        )


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        print("\nRESULT")

        print(
            "Chunk size:",
            chunk_size
        )

        print(
            "Overlap:",
            overlap
        )

        print(
            "Total chunks:",
            total_chunks
        )

        print(
            f"Recall@1: "
            f"{recall_1_percent:.2f} %"
        )

        print(
            f"Recall@3: "
            f"{recall_3_percent:.2f} %"
        )

        print(
            f"Recall@5: "
            f"{recall_5_percent:.2f} %"
        )

        print(
            f"MRR: "
            f"{mrr:.4f}"
        )


        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        all_results.append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_chunks": total_chunks,
                "recall_at_1": recall_1_percent,
                "recall_at_3": recall_3_percent,
                "recall_at_5": recall_5_percent,
                "mrr": mrr
            }
        )


# =========================================================
# FINAL COMPARISON
# =========================================================

print("\n\n")
print("==========================================")
print("FINAL CHUNKING COMPARISON")
print("==========================================")


print(
    "\nChunk\tOverlap\tR@1\tR@3\tR@5\tMRR"
)


for result in all_results:

    print(
        f"{result['chunk_size']}\t"
        f"{result['overlap']}\t"
        f"{result['recall_at_1']:.2f}\t"
        f"{result['recall_at_3']:.2f}\t"
        f"{result['recall_at_5']:.2f}\t"
        f"{result['mrr']:.4f}"
    )


# =========================================================
# FIND BEST CONFIGURATION
# =========================================================

best_result = max(
    all_results,
    key=lambda x: (
        x["recall_at_5"],
        x["mrr"],
        x["recall_at_3"]
    )
)


print("\n")
print("==========================================")
print("BEST CONFIGURATION")
print("==========================================")


print(
    "Chunk size:",
    best_result["chunk_size"]
)

print(
    "Overlap:",
    best_result["overlap"]
)

print(
    f"Recall@1: "
    f"{best_result['recall_at_1']:.2f} %"
)

print(
    f"Recall@3: "
    f"{best_result['recall_at_3']:.2f} %"
)

print(
    f"Recall@5: "
    f"{best_result['recall_at_5']:.2f} %"
)

print(
    f"MRR: "
    f"{best_result['mrr']:.4f}"
)


print("\nExperiment completed.")