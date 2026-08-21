import json
import csv

INPUT_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\sample_records.jsonl"

OUTPUT_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\bilingual_passages.csv"


print("=" * 70)
print("CREATING BILINGUAL PASSAGE DATASET")
print("=" * 70)

rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line_number, line in enumerate(f, start=1):

        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print("Skipping invalid JSON line:", line_number)
            print(e)
            continue

        query_id = record.get("query_id")
        query = record.get("query", "")
        english_query = record.get("Eng_Query", "")
        english_answer = record.get("Eng_Answer", "")

        passages = record.get("passages", {})

        english_passages = passages.get(
            "English_passages",
            []
        )

        translated_passages = passages.get(
            "Translated_passages",
            []
        )

        selected = passages.get(
            "is_selected",
            []
        )

        count = min(
            len(english_passages),
            len(translated_passages),
            len(selected)
        )

        for i in range(count):

            rows.append({
                "query_id": query_id,
                "query": query,
                "english_query": english_query,
                "english_answer": english_answer,
                "passage_id": i,
                "english_passage": english_passages[i],
                "assamese_passage": translated_passages[i],
                "is_selected": selected[i]
            })


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [
    "query_id",
    "query",
    "english_query",
    "english_answer",
    "passage_id",
    "english_passage",
    "assamese_passage",
    "is_selected"
]

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

print("\nDataset created successfully.")

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nTotal rows:", len(rows))

print(
    "Selected passages:",
    sum(
        1
        for row in rows
        if str(row["is_selected"]) == "1"
    )
)

print(
    "Unique queries:",
    len(
        set(
            row["query_id"]
            for row in rows
        )
    )
)

print("\nColumns:")
print(fieldnames)

print("\n")
print("=" * 70)
print("COMPLETED")
print("=" * 70)