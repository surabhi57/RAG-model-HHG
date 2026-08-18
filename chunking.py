def chunk_fixed(text, chunk_size=80, overlap=20, min_words=15):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i+chunk_size]
        # skip tiny trailing fragments that add noise without useful meaning
        if len(chunk_words) < min_words:
            continue
        chunk = " ".join(chunk_words)
        if chunk:
            chunks.append(chunk)
    return chunks
