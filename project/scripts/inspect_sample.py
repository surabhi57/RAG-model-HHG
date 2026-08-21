import json

FILE = "sample_records.jsonl"

print("Reading sample_records.jsonl...")

total_records = 0
records_with_selected = 0
records_with_relevant = 0

with open(FILE, "r", encoding="utf-8") as file:

    for line in file:

        if not line.strip():
            continue

        record = json.loads(line)

        total_records += 1

        # Look for selected information
        selected = record.get("selected", [])

        if selected:
            records_with_selected += 1

            if 1 in selected:
                records_with_relevant += 1


        if total_records == 1:

            print("\nFirst record keys:")
            print(record.keys())

            print("\nFirst record:")
            print(record)


print("\n==========================================")
print("DATASET SUMMARY")
print("==========================================")

print("Total records:", total_records)

print(
    "Records with selected field:",
    records_with_selected
)

print(
    "Records containing selected=1:",
    records_with_relevant
)

print("==========================================")