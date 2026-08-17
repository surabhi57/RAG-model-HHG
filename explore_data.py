import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="validation/hinval.parquet",
    repo_type="dataset"
)

df = pd.read_parquet(path)
row = df.iloc[0]

print("Query:", row["query"])
print("Eng Query:", row["Eng_Query"])
print()
print("Passages type:", type(row["passages"]))
print("Passages keys:", list(row["passages"].keys()))
print()
for key in row["passages"].keys():
    val = row["passages"][key]
    print(f"Key: {key} | type: {type(val)} | length: {len(val) if hasattr(val, '__len__') else 'n/a'}")
print()
print("Full passages dict:")
print(row["passages"])