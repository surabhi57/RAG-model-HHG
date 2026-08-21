import re


def chunk_fixed(text, chunk_size=100, overlap=0):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_sentence(text, sentences_per_chunk=3):
    sentences = re.split(
        r'(?<=[.!?।])\s+',
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    for i in range(
        0,
        len(sentences),
        sentences_per_chunk
    ):
        chunk = " ".join(
            sentences[i:i + sentences_per_chunk]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_paragraph(text):
    paragraphs = re.split(
        r'\n\s*\n',
        text
    )

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks


def chunk_metadata(
    text,
    doc_id,
    language="hi",
    chunk_size=100,
    overlap=0
):
    words = text.split()
    chunks = []

    step = chunk_size - overlap
    position = 0

    for i in range(0, len(words), step):

        chunk_text = " ".join(
            words[i:i + chunk_size]
        )

        if not chunk_text:
            continue

        chunk = {
            "text": chunk_text,
            "doc_id": doc_id,
            "position": position,
            "language": language
        }

        chunks.append(chunk)
        position += 1

    return chunks