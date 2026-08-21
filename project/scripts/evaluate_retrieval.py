import csv


PASSAGES_FILE = "passages.csv"
QUERIES_FILE = "test_queries.csv"

TOP_K = 5


print("==========================================")
print("RETRIEVAL EVALUATION STARTED")
print("==========================================")


# ---------------------------------------------------------
# LOAD PASSAGES
# ---------------------------------------------------------

passages = []

print("\nReading passages.csv...")

with open(
    PASSAGES_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    print("Passage columns:")
    print(reader.fieldnames)

    for row in reader:

        passage = row["passage"].strip()

        if passage:

            passages.append({
                "passage_id": row["passage_id"],
                "text": passage
            })


print("Total passages loaded:", len(passages))


# ---------------------------------------------------------
# LOAD QUERIES
# ---------------------------------------------------------

queries = []

print("\nReading test_queries.csv...")

with open(
    QUERIES_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    print("Query columns:")
    print(reader.fieldnames)

    for row in reader:

        queries.append(row)


print("Total queries loaded:", len(queries))


# ---------------------------------------------------------
# RETRIEVAL FUNCTION
# ---------------------------------------------------------

def retrieve(query, passages, top_k=5):

    query_words = set(
        query.lower().split()
    )

    scored_passages = []

    for passage in passages:

        passage_words = set(
            passage["text"].lower().split()
        )

        score = len(
            query_words.intersection(
                passage_words
            )
        )

        scored_passages.append(
            (
                score,
                passage
            )
        )

    scored_passages.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored_passages[:top_k]


# ---------------------------------------------------------
# EVALUATE EACH QUERY
# ---------------------------------------------------------

correct = 0

print("\n==========================================")
print("QUERY RESULTS")
print("==========================================")


for number, item in enumerate(
    queries,
    start=1
):

    query = item["query"]

    expected_id = item[
        "expected_passage_id"
    ]

    results = retrieve(
        query,
        passages,
        TOP_K
    )

    retrieved_ids = []

    for score, passage in results:

        retrieved_ids.append(
            passage["passage_id"]
        )


    found = expected_id in retrieved_ids


    if found:

        correct += 1


    print("\n------------------------------------------")

    print("Query", number)

    print("Query:", query)

    print(
        "Expected passage ID:",
        expected_id
    )

    print(
        "Retrieved passage IDs:",
        retrieved_ids
    )

    print(
        "Correct:",
        found
    )


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

total = len(queries)


if total > 0:

    accuracy = (
        correct / total
    ) * 100

else:

    accuracy = 0


print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

print("Correct:", correct)

print("Total:", total)

print(
    "Top-5 Retrieval Accuracy:",
    round(accuracy, 2),
    "%"
)

print("==========================================")python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"