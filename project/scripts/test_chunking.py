from chunking import (
    chunk_fixed,
    chunk_sentence,
    chunk_paragraph,
    chunk_metadata
)


text = """
Artificial intelligence is changing many areas of technology.
Machine learning allows computers to learn from data.
Natural language processing helps computers understand human language.

Retrieval augmented generation combines search with language models.
This can improve the accuracy of AI systems.
"""


print("=" * 60)
print("1. FIXED CHUNKING")
print("=" * 60)

chunks = chunk_fixed(
    text,
    chunk_size=10,
    overlap=2
)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(chunk)


print("\n")
print("=" * 60)
print("2. SENTENCE CHUNKING")
print("=" * 60)

chunks = chunk_sentence(
    text,
    sentences_per_chunk=2
)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(chunk)


print("\n")
print("=" * 60)
print("3. PARAGRAPH CHUNKING")
print("=" * 60)

chunks = chunk_paragraph(text)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(chunk)


print("\n")
print("=" * 60)
print("4. METADATA-AWARE CHUNKING")
print("=" * 60)

chunks = chunk_metadata(
    text,
    doc_id="TEST_001",
    language="hi",
    chunk_size=10,
    overlap=2
)

print("Number of chunks:", len(chunks))

for chunk in chunks:

    print("\nChunk:")
    print("Text:", chunk["text"])
    print("Document ID:", chunk["doc_id"])
    print("Position:", chunk["position"])
    print("Language:", chunk["language"])