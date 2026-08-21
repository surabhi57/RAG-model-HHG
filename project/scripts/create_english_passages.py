import json
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\sample_records.jsonl"

OUTPUT_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\english_passages.csv"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CREATING ENGLISH PASSAGES CSV")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)

print("\nOutput file:")
print(OUTPUT_FILE)


# ============================================================
# READ JSONL
# ============================================================

records = []

print("\nReading sample_records.jsonl...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        records.append(record)


print("Records loaded:", len(records))


# ============================================================
# EXTRACT ENGLISH PASSAGES
# ============================================================

rows = []

for record in records:

    query_id = record.get("query_id")

    english_query = record.get("Eng_Query", "")

    english_passages = (
        record.get("passages", {})
        .get("English_passages", [])
    )

    selected = (
        record.get("passages", {})
        .get("is_selected", [])
    )


    for i, passage in enumerate(english_passages):

        if not passage:
            continue

        is_selected = 0

        if i < len(selected):
            is_selected = selected[i]


        rows.append(
            {
                "query_id": query_id,
                "query": english_query,
                "passage_id": i,
                "passage": passage,
                "is_selected": is_selected
            }
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


print("\nEnglish passages extracted:", len(df))


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nEnglish passages CSV created successfully.")

print("Rows:", len(df))

print("Columns:")
print(list(df.columns))

print("\nSaved at:")
print(OUTPUT_FILE)


# ============================================================
# PREVIEW
# ============================================================

print("\nFirst 5 rows:")

print(
    df.head().to_string(index=False)
)

print("\n" + "=" * 70)