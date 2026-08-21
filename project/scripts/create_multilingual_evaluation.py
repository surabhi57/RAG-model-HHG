import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Users\rrutu\OneDrive\Desktop\project\scripts"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "bilingual_passages.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "evaluation_queries.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CREATING MULTILINGUAL EVALUATION DATASET")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading bilingual_passages.csv...")

df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)

print(
    "Total rows:",
    len(df)
)

print(
    "Columns:",
    df.columns.tolist()
)


# ============================================================
# SELECT RELEVANT PASSAGES
# ============================================================

selected = df[
    df["is_selected"] == 1
].copy()

print(
    "\nSelected relevant rows:",
    len(selected)
)


# ============================================================
# CREATE ONE EVALUATION QUERY PER QUERY_ID
# ============================================================

evaluation_rows = []

for query_id, group in selected.groupby(
    "query_id"
):

    # Take the first selected passage
    row = group.iloc[0]

    evaluation_rows.append(
        {
            "query_id":
                query_id,

            "query":
                row["query"],

            "expected_passage_id":
                row["passage_id"],

            "expected_passage":
                row["english_passage"],

            "is_selected":
                row["is_selected"]
        }
    )


evaluation_df = pd.DataFrame(
    evaluation_rows
)


# ============================================================
# SAVE
# ============================================================

evaluation_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION DATASET CREATED")
print("=" * 70)

print(
    "\nNumber of unique queries:",
    len(evaluation_df)
)

print(
    "\nFirst 10 queries:"
)

print(
    evaluation_df.head(10).to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_CSV
)

print("\n" + "=" * 70)
print("DATASET CREATION COMPLETED")
print("=" * 70)