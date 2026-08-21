import csv


INPUT_FILE = "passages.csv"
OUTPUT_FILE = "test_queries_50.csv"

TARGET_QUERIES = 50


print("==========================================")
print("CREATING 50 QUERY EVALUATION SET")
print("==========================================")


queries = []
seen_queries = set()


# ---------------------------------------------------------
# READ PASSAGES
# ---------------------------------------------------------

print("\nReading passages.csv...")


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    print("Columns found:")
    print(reader.fieldnames)


    for row in reader:

        # Only use relevant passages

        if row["is_selected"].strip() != "1":
            continue


        query = row["query"].strip()


        if not query:
            continue


        # Avoid duplicate queries

        if query in seen_queries:
            continue


        seen_queries.add(query)


        queries.append({
            "query_id":
                row["query_id"],

            "query":
                query,

            "expected_passage_id":
                row["passage_id"],

            "expected_passage":
                row["passage"],

            "is_selected":
                row["is_selected"]
        })


        if len(queries) >= TARGET_QUERIES:
            break


# ---------------------------------------------------------
# CHECK RESULT
# ---------------------------------------------------------

print(
    "\nRelevant unique queries found:",
    len(queries)
)


if len(queries) < TARGET_QUERIES:

    print(
        "WARNING: Only",
        len(queries),
        "queries were found."
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

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


print(
    "\nSaved file:",
    OUTPUT_FILE
)


print(
    "Number of queries:",
    len(queries)
)


print("\n==========================================")
print("DONE")
print("==========================================")