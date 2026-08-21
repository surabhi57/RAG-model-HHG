import json
import csv


INPUT_FILE = "sample_records.jsonl"
OUTPUT_FILE = "extracted_queries.csv"


print("==========================================")
print("EXTRACTING RELEVANT QUERIES")
print("==========================================")


results = []


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line_number, line in enumerate(
        file,
        start=1
    ):

        if not line.strip():
            continue


        record = json.loads(line)


        # Get query

        query = record.get(
            "query",
            ""
        )


        # Get passages object

        passages = record.get(
            "passages",
            {}
        )


        # Get translated passages

        translated_passages = passages.get(
            "Translated_passages",
            []
        )


        # Get relevance labels

        selected = passages.get(
            "is_selected",
            []
        )


        # Find the relevant passage

        relevant_index = None


        for i, value in enumerate(
            selected
        ):

            if int(value) == 1:

                relevant_index = i

                break


        # Save if a relevant passage exists

        if (
            query
            and relevant_index is not None
            and relevant_index < len(
                translated_passages
            )
        ):

            results.append({

                "query_id":
                    record.get(
                        "query_id",
                        line_number
                    ),

                "query":
                    query,

                "expected_passage_id":
                    relevant_index,

                "expected_passage":
                    translated_passages[
                        relevant_index
                    ],

                "is_selected":
                    1
            })


print()
print(
    "Relevant queries found:",
    len(results)
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


    writer.writerows(
        results
    )


print(
    "Saved file:",
    OUTPUT_FILE
)


print(
    "Number of queries:",
    len(results)
)


print("\n==========================================")
print("EXTRACTION COMPLETED")
print("==========================================")