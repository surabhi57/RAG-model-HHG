import json
import pandas as pd
import re

print("PREPARE_PASSAGES SCRIPT STARTED", flush=True)

# -----------------------------------------
# 1. Input file
# -----------------------------------------

input_file = "sample_records.jsonl"

print(
    f"Opening local file: {input_file}",
    flush=True
)

records = []

# -----------------------------------------
# 2. Read the local JSONL file
# -----------------------------------------

with open(
    input_file,
    "r",
    encoding="utf-8"
) as f:

    for record_number, line in enumerate(f):

        if not line.strip():
            continue

        sample = json.loads(line)

        # ---------------------------------
        # 3. Get query information
        # ---------------------------------

        query_id = sample.get(
            "query_id"
        )

        query = sample.get(
            "query",
            ""
        )

        # ---------------------------------
        # 4. Get passage information
        # ---------------------------------

        passages = sample.get(
            "passages",
            {}
        )

        translated_passages = passages.get(
            "Translated_passages",
            []
        )

        selected_labels = passages.get(
            "is_selected",
            []
        )

        # ---------------------------------
        # 5. Process each passage
        # ---------------------------------

        for passage_number, passage in enumerate(
            translated_passages
        ):

            if not passage:
                continue

            # Convert to string
            passage = str(passage)

            # Remove extra spaces/newlines
            passage = re.sub(
                r"\s+",
                " ",
                passage
            ).strip()

            # Skip empty passages
            if not passage:
                continue

            # Get relevance label
            if passage_number < len(
                selected_labels
            ):
                selected = selected_labels[
                    passage_number
                ]
            else:
                selected = 0

            # Store the information
            records.append({

                "query_id": query_id,

                "query": query,

                "passage_id": passage_number,

                "passage": passage,

                "is_selected": selected

            })

        print(
            f"Processed record {record_number + 1}",
            flush=True
        )

# -----------------------------------------
# 6. Create DataFrame
# -----------------------------------------

print(
    "\nFinished reading local dataset!",
    flush=True
)

df = pd.DataFrame(records)

print(
    f"Total passages collected: {len(df)}",
    flush=True
)

# -----------------------------------------
# 7. Save CSV
# -----------------------------------------

output_file = "passages.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Saved successfully: {output_file}",
    flush=True
)

# -----------------------------------------
# 8. Display information
# -----------------------------------------

print("\nColumns:")

print(
    df.columns.tolist()
)

print("\nFirst 5 passages:")

if len(df) > 0:

    print(
        df.head().to_string()
    )

else:

    print(
        "WARNING: No passages were found."
    )