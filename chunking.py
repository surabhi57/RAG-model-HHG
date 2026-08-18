import re

def chunk_fixed(text, chunk_size=80, overlap=20, min_words=15):
    """Fixed-size chunking with overlap between consecutive chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) < min_words:
            continue
        chunk = " ".join(chunk_words)
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_fixed_no_overlap(text, chunk_size=80, min_words=15):
    """Fixed-size chunking with NO overlap - simplest baseline strategy."""
    return chunk_fixed(text, chunk_size=chunk_size, overlap=0, min_words=min_words)


def chunk_sentence_aware(text, max_chunk_words=80, min_words=15):
    """
    Splits text at sentence boundaries (Hindi and English punctuation),
    then greedily groups sentences together until close to max_chunk_words.
    Never cuts a sentence in half.
    """
    # split on Hindi danda (।), period, question mark, exclamation
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk_words = []

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        current_word_count = len(current_chunk_words)

        if current_word_count + sentence_word_count > max_chunk_words and current_chunk_words:
            # current chunk is full, save it and start a new one
            chunk_text = " ".join(current_chunk_words)
            if len(current_chunk_words) >= min_words:
                chunks.append(chunk_text)
            current_chunk_words = sentence.split()
        else:
            current_chunk_words.extend(sentence.split())

    # don't forget the last chunk
    if current_chunk_words and len(current_chunk_words) >= min_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


if __name__ == "__main__":
    # quick sanity test comparing all 3 strategies on one example
    sample_text = ("एक कंपनी एक विशिष्ट देश में निगमित होती है, अक्सर उस देश के एक छोटे उपसमूह, "
                    "जैसे कि एक राज्य या प्रांत, की सीमाओं के भीतर। निगम तब उस राज्य में निगमन के "
                    "कानूनों द्वारा शासित होता है। एक निगम या तो निजी या सार्वजनिक स्टॉक जारी कर "
                    "सकता है, या इसे गैर-स्टॉक निगम के रूप में वर्गीकृत किया जा सकता है।")

    print("--- Fixed with overlap ---")
    for c in chunk_fixed(sample_text, chunk_size=20, overlap=5, min_words=5):
        print(f"  [{len(c.split())} words]", c[:80])

    print("\n--- Fixed no overlap ---")
    for c in chunk_fixed_no_overlap(sample_text, chunk_size=20, min_words=5):
        print(f"  [{len(c.split())} words]", c[:80])

    print("\n--- Sentence aware ---")
    for c in chunk_sentence_aware(sample_text, max_chunk_words=20, min_words=5):
        print(f"  [{len(c.split())} words]", c[:80])
