import openai
import os
from dotenv import load_dotenv
from rag_app.retrieval import retrieve

load_dotenv()  # Loads .env variables

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def refine_answer(query, context, initial_answer, iterations=1):
    answer = initial_answer
    for i in range(iterations):
        print(f"[DEBUG] Running refine_answer() iteration {i+1} in llm_call.py")
        print(f"[DEBUG] Top 5 items of answer before refinement: {answer[:5]}")
        critique_prompt = f"""CONTEXT:
Context: {context}

Question: {query}
Initial Answer: {answer}

Refine the above answer for clarity, depth, and evidence. If possible, add more analysis or support from the context. Output only the improved answer."""
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a literary analyst. Your job is to refine and improve your previous answer for clarity, depth, and evidence, using the provided context."},
                {"role": "user", "content": critique_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        print(f"[DEBUG] Top 5 items of answer after refinement: {answer[:5]}")
    return answer

def answer(query):
    print("[DEBUG] Running answer() in llm_call.py")
    context = retrieve(query)
    # Truncate context if too long (e.g., 4000 characters)
    if len(context) > 4000:
        print(f"[DEBUG] Context too long ({len(context)} chars), truncating.")
        context = context[:4000]
    print(f"[DEBUG] Top 5 items of context: {context[:5]}")
    prompt = f"""CONTEXT:
Context: {context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": """You are a literary analyst focused on psychological character analysis.
                                            You are given multiple excerpts from a novel. These passages may come from different chapters or points in time.
                                            Your goal is to infer and explain the main character's mindset, emotional state, patterns of behavior, and internal conflicts — based on what is revealed across all passages.
                                            Carefully examine each excerpt before answering. Look for consistent emotional cues, contradictions, and evolving beliefs. If the character expresses confusion, anger, affection, or guilt, explore the underlying reasons and how they connect across time.
                                            Make thoughtful, evidence-based inferences. Do not summarize — analyze.
                                            Do not say "the context is vague" unless you have deeply considered every snippet. Reason step by step and support your claims with examples from the text. Keep your responses concise, around 250 tokens
             """},
            {"role": "user", "content": prompt}
        ], 
        max_tokens=250,
        temperature=0.7
    )
    initial_answer = response.choices[0].message.content.strip()
    print(f"[DEBUG] Top 5 items of initial_answer: {initial_answer[:5]}")
    # Self-refinement loop: 1 iteration (can increase if desired)
    refined = refine_answer(query, context, initial_answer, iterations=1)
    print(f"[DEBUG] Top 5 items of refined: {refined[:5]}")
    return refined