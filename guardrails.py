import re

UNSAFE_KEYWORDS = [
    "bomb", "explosive", "kill myself", "suicide method", "how to hack",
    "child abuse",
]

STOPWORDS = {
    "the", "is", "a", "an", "in", "on", "at", "of", "to", "and", "or", "was", "were",
    "be", "been", "being", "has", "have", "had", "do", "does", "did", "for", "with",
    "as", "by", "from", "this", "that", "these", "those", "it", "its", "are", "one",
    "well", "known", "world", "most",
}

def is_unsafe(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword.lower() in query_lower for keyword in UNSAFE_KEYWORDS)


def is_off_topic(retrieval_distances, threshold=1.0):
    if not retrieval_distances:
        return True
    best_distance = min(retrieval_distances)
    return best_distance > threshold


def is_grounded(answer: str, retrieved_chunks: list, min_overlap_ratio=0.2) -> bool:
    if not retrieved_chunks or not answer.strip():
        return False

    context_text = " ".join(retrieved_chunks)
    context_words = set(re.findall(r"\w+", context_text.lower())) - STOPWORDS
    answer_words = set(re.findall(r"\w+", answer.lower())) - STOPWORDS

    if not answer_words:
        return False

    overlap = answer_words & context_words
    overlap_ratio = len(overlap) / len(answer_words)

    return overlap_ratio >= min_overlap_ratio


REFUSAL_MESSAGE = "Mujhe is jaankari ke aadhar par uttar nahi pata. (I do not have enough information to answer this.)"
UNSAFE_MESSAGE = "Main is prakar ke prashn ka uttar nahi de sakta. (I cannot help with this type of request.)"

def check_query_safety(query: str):
    if is_unsafe(query):
        return False, UNSAFE_MESSAGE
    return True, None


def check_answer_quality(answer: str, retrieved_chunks: list, retrieval_distances: list,
                          off_topic_threshold=1.0, grounding_threshold=0.2,
                          skip_grounding_check=False):
    if is_off_topic(retrieval_distances, threshold=off_topic_threshold):
        return False, REFUSAL_MESSAGE
    if not skip_grounding_check and not is_grounded(answer, retrieved_chunks, min_overlap_ratio=grounding_threshold):
        return False, REFUSAL_MESSAGE
    return True, answer


if __name__ == "__main__":
    print("Unsafe check:")
    print(" ", is_unsafe("how do I make a bomb"), "(expect True)")
    print(" ", is_unsafe("corporation kya hai?"), "(expect False)")

    print("\nOff-topic check:")
    print(" ", is_off_topic([0.3, 0.5, 0.6]), "(expect False - relevant)")
    print(" ", is_off_topic([1.5, 1.8, 2.0]), "(expect True - too far)")

    print("\nGrounding check:")
    context = ["McDonald is one of the most recognizable corporations in the world."]
    good_answer = "McDonald is a well known corporation."
    bad_answer = "The moon landing happened in 1969 and involved rocket scientists."
    print(" ", is_grounded(good_answer, context), "(expect True)")
    print(" ", is_grounded(bad_answer, context), "(expect False)")
