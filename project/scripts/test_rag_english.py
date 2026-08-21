import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

question = "What is a corporation?"

context = """
McDonald's Corporation is one of the most recognizable corporations in the world.
A corporation is a company or group of people authorized to act as a single entity
(legally a person) and recognized as such in law.
Early incorporated entities were established by charter.
"""

prompt = f"""
Answer the question using the context.

Question:
{question}

Context:
{context}

Answer:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4,
        do_sample=False
    )

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
).strip()

print("=" * 60)
print("RAG ENGLISH GENERATION TEST")
print("=" * 60)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)