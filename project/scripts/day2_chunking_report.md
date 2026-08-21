# Day 2 – Chunking Strategy Research

## Objective

The objective of this experiment was to study different text chunking
strategies and understand how they affect the number and size of chunks
generated from the MSMARCO-XI dataset.

## Dataset

Dataset: AI4Bharat MSMARCO-XI

Language: Hindi

Number of passages tested: 50

The passages were obtained from the prepared `passages.csv` file.

## Chunking Strategies

### 1. Fixed-size Chunking

The passage is divided into chunks containing a fixed number of words.

In this experiment:

- Chunk size = 100 words
- Overlap = 0 words

### 2. Sentence-based Chunking

The text is divided according to sentence boundaries.

In this experiment:

- 3 sentences were grouped into one chunk.

### 3. Paragraph-based Chunking

The text is divided using paragraph boundaries.

This approach attempts to preserve larger contextual units.

### 4. Metadata-aware Chunking

This strategy stores additional information with every chunk.

Each chunk contains:

- Chunk text
- Document ID
- Position of the chunk
- Language

Example:

```text
Text: <chunk text>
Document ID: <document ID>
Position: 0
Language: hi