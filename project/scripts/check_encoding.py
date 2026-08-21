import csv


CSV_FILE = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts\passages.csv"


print("==========================================")
print("ENCODING CHECK")
print("==========================================")


with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


print(
    "Rows:",
    len(rows)
)


print("\nFirst 5 queries:\n")


for i, row in enumerate(rows[:5], start=1):

    print(
        f"Query {i}:"
    )

    print(
        row["query"]
    )

    print()


print("==========================================")
print("CHECKING FOR MOJIBAKE")
print("==========================================")


bad_patterns = [
    "à¦",
    "à§",
    "Ã",
    "Â",
    "ð",
    "�"
]


bad_count = 0


for row in rows:

    text = (
        row["query"]
        + " "
        + row["passage"]
    )


    if any(
        pattern in text
        for pattern in bad_patterns
    ):

        bad_count += 1


print(
    "Rows containing suspicious encoding:",
    bad_count
)


print("==========================================")
