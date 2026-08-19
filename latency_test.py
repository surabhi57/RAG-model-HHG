import json
import numpy as np
import pandas as pd
import time
from huggingface_hub import hf_hub_download

from pipeline import run_pipeline, collection

NUM_TEST_QUERIES = 30

path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="validation/hinval.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(path)

test_queries = df.iloc[:NUM_TEST_QUERIES]["query"].tolist()

print(f"Running {len(test_queries)} test queries through the full pipeline...\n")

results = []
for i, q in enumerate(test_queries):
    print(f"[{i+1}/{len(test_queries)}] {q}")
    time.sleep(4)
    try:
        result = run_pipeline(q)
        results.append(result)
    except Exception as e:
        print(f"  FAILED: {e}")

def percentiles(values, label):
    arr = np.array(values)
    p50 = np.percentile(arr, 50)
    p70 = np.percentile(arr, 70)
    p100 = np.percentile(arr, 100)
    print(f"\n{label}")
    print(f"  P50:  {p50:.2f} ms")
    print(f"  P70:  {p70:.2f} ms")
    print(f"  P100: {p100:.2f} ms (max)")
    return {"p50": round(p50, 2), "p70": round(p70, 2), "p100": round(p100, 2)}

retrieval_times = [r["timings"]["retrieval_ms"] for r in results]
generation_times = [r["timings"]["generation_ms"] for r in results]
total_times = [r["timings"]["total_ms"] for r in results]

print("\n" + "=" * 50)
print(f"LATENCY REPORT ({len(results)} successful queries)")
print("=" * 50)

retrieval_stats = percentiles(retrieval_times, "Retrieval only (chunking + vector search)")
generation_stats = percentiles(generation_times, "Generation only (LLM call)")
total_stats = percentiles(total_times, "End-to-end total")

report = {
    "num_queries": len(results),
    "retrieval_ms": retrieval_stats,
    "generation_ms": generation_stats,
    "total_ms": total_stats,
    "raw_results": [
        {
            "query": r["query"],
            "answer": r["answer"],
            "timings": r["timings"],
            "top_chunk_is_selected": r["retrieved_metadata"][0]["is_selected"]
        }
        for r in results
    ]
}

with open("latency_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\nFull report saved to latency_report.json")

correct_retrievals = sum(1 for r in results if r["retrieved_metadata"][0]["is_selected"] == 1)
print(f"\nRetrieval quality: top-1 result was the correct passage in {correct_retrievals}/{len(results)} queries ({100*correct_retrievals/len(results):.1f}%)")
