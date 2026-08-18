from pipeline import retrieve, generate_answer, collection
from guardrails import is_off_topic, is_grounded

query = "\u0915\u0949\u0930\u094d\u092a\u094b\u0930\u0947\u0936\u0928 \u0915\u094d\u092f\u093e \u0939\u0948?"

docs, metas, distances = retrieve(query, k=3)

print("Query:", query)
print("\nRetrieved chunks and distances:")
for doc, dist in zip(docs, distances):
    print(f"  distance={dist:.4f} | {doc[:80]}")

print("\nis_off_topic result:", is_off_topic(distances, threshold=1.0))
print("Min distance:", min(distances))

answer = generate_answer(query, docs)
print("\nGenerated answer:", answer)

grounded = is_grounded(answer, docs, min_overlap_ratio=0.2)
print("\nis_grounded result:", grounded)

import re
STOPWORDS = {
    "the", "is", "a", "an", "in", "on", "at", "of", "to", "and", "or", "was", "were",
    "be", "been", "being", "has", "have", "had", "do", "does", "did", "for", "with",
    "as", "by", "from", "this", "that", "these", "those", "it", "its", "are", "one",
    "well", "known", "world", "most",
}
context_text = " ".join(docs)
context_words = set(re.findall(r"\w+", context_text.lower())) - STOPWORDS
answer_words = set(re.findall(r"\w+", answer.lower())) - STOPWORDS
overlap = answer_words & context_words
print("\nAnswer words:", answer_words)
print("Overlap words:", overlap)
print("Overlap ratio:", len(overlap) / len(answer_words) if answer_words else 0)
