import pandas as pd
from huggingface_hub import hf_hub_download
from pipeline import retrieve, collection

NUM_QUERIES = 30

path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="validation/hinval.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(path)
df_sample = df.iloc[:NUM_QUERIES]

correct_match_distances = []
incorrect_match_distances = []

for _, row in df_sample.iterrows():
    query = row["query"]
    is_selected = list(row["passages"]["is_selected"])
    if 1 not in is_selected:
        continue

    docs, metas, distances = retrieve(query, k=3)
    top1_correct = metas[0]["is_selected"] == 1

    if top1_correct:
        correct_match_distances.append(distances[0])
    else:
        incorrect_match_distances.append(distances[0])

print(f"Correct top-1 matches ({len(correct_match_distances)} queries):")
if correct_match_distances:
    print(f"  min={min(correct_match_distances):.2f}  max={max(correct_match_distances):.2f}  avg={sum(correct_match_distances)/len(correct_match_distances):.2f}")

print(f"\nIncorrect top-1 matches ({len(incorrect_match_distances)} queries):")
if incorrect_match_distances:
    print(f"  min={min(incorrect_match_distances):.2f}  max={max(incorrect_match_distances):.2f}  avg={sum(incorrect_match_distances)/len(incorrect_match_distances):.2f}")

print("\nAll distances sorted (correct=C, incorrect=I):")
labeled = [(d, "C") for d in correct_match_distances] + [(d, "I") for d in incorrect_match_distances]
labeled.sort()
for d, label in labeled:
    print(f"  {d:.2f}  {label}")
