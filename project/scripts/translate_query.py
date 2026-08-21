from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ============================================================
# ASSAMESE → ENGLISH QUERY TRANSLATION
# ============================================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"


print("=" * 70)
print("ASSAMESE → ENGLISH QUERY TRANSLATOR")
print("=" * 70)

print("\nLoading translation model...")
print("Model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("\nTranslation model loaded successfully.")


# ------------------------------------------------------------
# Translation function
# ------------------------------------------------------------

def translate_assamese_to_english(text):

    tokenizer.src_lang = "asm_Beng"

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=
            tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128
        )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


# ------------------------------------------------------------
# Test query
# ------------------------------------------------------------

assamese_query = "কৰ্পোৰেচন কি?"

print("\n" + "=" * 70)
print("TRANSLATION TEST")
print("=" * 70)

print("\nAssamese Question:")
print(assamese_query)

english_query = translate_assamese_to_english(
    assamese_query
)

print("\nEnglish Translation:")
print(english_query)

print("\n" + "=" * 70)
print("TRANSLATION TEST COMPLETED")
print("=" * 70)