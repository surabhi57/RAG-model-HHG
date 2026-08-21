import csv

INPUT_FILE = "passages.csv"
OUTPUT_FILE = "test_queries.csv"

NUMBER_OF_QUERIES = 20

queries = []

seen_queries = set()

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        query = row["query"].strip()

        if not query:
            continue

        # Only use relevant passages
        if row["is_selected"].strip() != "1":
            continue

        # Avoid duplicate queries
        if query in seen_queries:
            continue

        seen_queries.add(query)

        queries.append({
            "query_id": row["query_id"],
            "query": query,
            "expected_passage_id": row["passage_id"],
            "expected_passage": row["passage"],
            "is_selected": row["is_selected"]
        })

        if len(queries) == NUMBER_OF_QUERIES:
            break


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    fieldnames = [
        "query_id",
        "query",
        "expected_passage_id",
        "expected_passage",
        "is_selected"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(queries)


print("Test query creation completed.")
print("Number of relevant queries:", len(queries))
print("Saved file:", OUTPUT_FILE)