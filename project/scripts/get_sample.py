from huggingface_hub import hf_hub_download
import json
import pandas as pd
import re

print("Downloading Hindi validation data...")

file_path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="validation/hinval.jsonl",
    repo_type="dataset"
)

print("Dataset file ready!")

records = []

print("\nReading records...")

with open(file_path, "r", encoding="utf-8") as f:

    for record_number, line in enumerate(f):

        if record_number >= 50:
            break

        sample = json.loads(line)

        query = sample.get("query")
        query_id = sample.get("query_id")

        passages = sample.get("passages", {})

        translated_passages = passages.get(
            "Translated_passages",
            []
        )

        selected_labels = passages.get(
            "is_selected",
            []
        )

        for passage_number, passage in enumerate(
            translated_passages
        ):

            # Skip missing passages
            if passage is None:
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
            if passage == "":
                continue

            # Get relevance label
            if passage_number < len(selected_labels):
                is_selected = selected_labels[passage_number]
            else:
                is_selected = 0

            records.append({
                "query_id": query_id,
                "query": query,
                "passage_id": passage_number,
                "passage": passage,
                "is_selected": is_selected
            })

print("\nTotal passages collected:", len(records))

df = pd.DataFrame(records)

output_file = "passages.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", output_file)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows: