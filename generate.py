import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def generate_answer(query, chunks):
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    prompt = "You are a helpful assistant answering questions using ONLY the context provided below.\n\nRules:\n- Answer only using information from the context. Do not use outside knowledge.\n- If the context does not contain enough information to answer, respond exactly with: \"Mujhe is jaankari ke aadhar par uttar nahi pata.\"\n- Keep the answer concise and in the same language as the question.\n\nContext:\n" + context + "\n\nQuestion: " + query + "\n\nAnswer:"
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text.strip()


if __name__ == "__main__":
    test_query = "corporation kya hai?"
    test_chunks = [
        "McDonald's Corporation is one of the most recognizable corporations in the world. A corporation is a company or group of people authorized to act as a single entity.",
        "Corporation definition: a group of people created by law, having continuous existence independent of its members."
    ]
    answer = generate_answer(test_query, test_chunks)
    print("Query:", test_query)
    print("Answer:", answer)
