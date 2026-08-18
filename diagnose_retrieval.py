import pandas as pd
from huggingface_hub import hf_hub_download

from pipeline import retrieve, collection

NUM_TEST_QUERIES = 30

path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="validation/hinval.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(path)
df_sample = df.iloc[:NUM_TEST_QUERIES]

correct_top1 = 0
correct_top3 = 0
evaluable_count = 0
skipped_no_ground_truth = 0

for _, row in df_sample.iterrows():
    query = row["query"]
    query_id = int(row["query_id"])

    passages_dict = row["passages"]
    hindi_passages = passages_dict["Translated_passages"]
    is_selected = list(passages_dict["is_selected"])

    if 1 not in is_selected:
        skipped_no_ground_truth += 1
        continue  # this query has no answerable passage in the dataset - skip it

    evaluable_count += 1
    correct_idx = is_selected.index(1)
    correct_passage = hindi_passages[correct_idx]

    docs, metas = retrieve(query, k=3)

    top1_is_selected = metas[0]["is_selected"]
    any_correct_in_top3 = any(m["is_selected"] == 1 for m in metas)

    if top1_is_selected == 1:
        correct_top1 += 1
    if any_correct_in_top3:
        correct_top3 += 1

    status = "TOP-1" if top1_is_selected == 1 else ("TOP-3" if any_correct_in_top3 else "MISS")
    print(f"[{status}] {query}")

print("\n" + "=" * 60)
print(f"Total queries checked: {NUM_TEST_QUERIES}")
print(f"Skipped (no ground truth answer in dataset): {skipped_no_ground_truth}")
print(f"Evaluable queries (have a real answer passage): {evaluable_count}")
print(f"\nTop-1 accuracy: {correct_top1}/{evaluable_count} ({100*correct_top1/evaluable_count:.1f}%)")
print(f"Top-3 accuracy: {correct_top3}/{evaluable_count} ({100*correct_top3/evaluable_count:.1f}%)")
