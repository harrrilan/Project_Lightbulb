import openai
import os
from dotenv import load_dotenv
from retrieval import retrieve

load_dotenv()  # Loads .env variables

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def answer(query):
    context = retrieve(query)
    prompt = f"""You are a helpful assistant.
Context: {context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",  # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a RAG assistant. You are given a context and a question. You need to answer the question based on the context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()