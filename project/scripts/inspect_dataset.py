from datasets import load_dataset
import time

print("Starting...", flush=True)

dataset = load_dataset(
    "ai4bharat/MSMARCO-XI",
    streaming=True
)

print("Dataset object created!", flush=True)

train = dataset["train"]

print("Attempting to read first record...", flush=True)
print("This may take some time because Hugging Face is streaming the shard.", flush=True)

start = time.time()

sample = next(iter(train))

elapsed = time.time() - start

print(f"\nFirst record received after {elapsed:.2f} seconds.", flush=True)

print("\nQuery:")
print(sample["query"], flush=True)

print("\nQuery ID:")
print(sample["query_id"], flush=True)

print("\nSource language:")
print(sample["source_lang"], flush=True)

print("\nTarget language:")
print(sample["target_lang"], flush=True)

print("\nNumber of translated passages:")
print(len(sample["passages"]["Translated_passages"]), flush=True)

print("\nSelected labels:")
print(sample["passages"]["is_selected"], flush=True)