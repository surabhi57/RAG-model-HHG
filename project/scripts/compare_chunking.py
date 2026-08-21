import csv
import statistics

from chunking import (
    chunk_fixed,
    chunk_sentence,
    chunk_paragraph,
    chunk_metadata
)


CSV_FILE = "passages.csv"
NUMBER_OF_PASSAGES = 50


# =========================================================
# READ 50 PASSAGES
# =========================================================

passages = []

with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    print("CSV columns found:")
    print(reader.fieldnames)

    for row in reader:

        passage = row["passage"].strip()

        if passage:

            passages.append({
                "query_id": row["query_id"],
                "passage_id": row["passage_id"],
                "query": row["query"],
                "passage": passage,
                "is_selected": row["is_selected"]
            })

        if len(passages) == NUMBER_OF_PASSAGES:
            break


print("\n" + "=" * 70)
print("CHUNKING COMPARISON")
print("=" * 70)

print("Number of passages loaded:", len(passages))


# =========================================================
# STORAGE
# =========================================================

results = {
    "Fixed": [],
    "Sentence": [],
    "Paragraph": [],
    "Metadata": []
}


# =========================================================
# PROCESS PASSAGES
# =========================================================

for item in passages:

    text = item["passage"]

    # 1. Fixed
    fixed_chunks = chunk_fixed(
        text,
        chunk_size=100,
        overlap=0
    )

    results["Fixed"].extend(fixed_chunks)


    # 2. Sentence
    sentence_chunks = chunk_sentence(
        text,
        sentences_per_chunk=3
    )

    results["Sentence"].extend(sentence_chunks)


    # 3. Paragraph
    paragraph_chunks = chunk_paragraph(text)

    results["Paragraph"].extend(paragraph_chunks)


    # 4. Metadata
    metadata_chunks = chunk_metadata(
        text,
        doc_id=item["passage_id"],
        language="hi",
        chunk_size=100,
        overlap=0
    )

    results["Metadata"].extend(metadata_chunks)


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 70)
print("RESULTS")
print("=" * 70)


for strategy, chunks in results.items():

    if not chunks:

        print("\n" + strategy + ": No chunks")

        continue


    if strategy == "Metadata":

        lengths = [
            len(chunk["text"].split())
            for chunk in chunks
        ]

    else:

        lengths = [
            len(chunk.split())
            for chunk in chunks
        ]


    average_length = statistics.mean(lengths)


    print("\nStrategy:", strategy)

    print("Number of chunks:", len(chunks))

    print(
        "Average chunk length:",
        round(average_length, 2),
        "words"
    )

    print(
        "Shortest chunk:",
        min(lengths),
        "words"
    )

    print(
        "Longest chunk:",
        max(lengths),
        "words"
    )


# =========================================================
# EXAMPLES
# =========================================================

print("\n")
print("=" * 70)
print("EXAMPLE CHUNKS")
print("=" * 70)


for strategy, chunks in results.items():

    print("\n###", strategy)

    for i, chunk in enumerate(chunks[:3]):

        print("\nExample", i + 1)

        if strategy == "Metadata":

            print("Text:", chunk["text"])
            print("Document ID:", chunk["doc_id"])
            print("Position:", chunk["position"])
            print("Language:", chunk["language"])

        else:

            print(chunk)