import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded.")

question = "What is a corporation?"

prompt = (
    "Answer this question briefly and clearly.\n"
    "Question: What is a corporation?\n"
    "Answer:"
)

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
        early_stopping=True
    )

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
).strip()

print("\n==========================================")
print("LLM TEST RESULT")
print("==========================================")
print("Answer:", repr(answer))