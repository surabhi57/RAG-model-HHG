import csv

rows = [
    ["Experiment", "Chunk Size", "Overlap", "Chunks", "Correct", "Total", "Top-5 Accuracy"],
    ["Chunk size", 50, 0, 252, 3, 6, "50%"],
    ["Chunk size", 100, 0, 201, 3, 6, "50%"],
    ["Chunk size", 200, 0, 200, 3, 6, "50%"],
    ["Overlap", 100, 0, 201, 3, 6, "50%"],
    ["Overlap", 100, 10, 202, 3, 6, "50%"],
    ["Overlap", 100, 20, 203, 3, 6, "50%"],
]

with open(
    "experiment_summary.csv",
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)
    writer.writerows(rows)

print("Experiment summary created.")
print("Saved file: experiment_summary.csv")