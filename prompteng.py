import openai
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()  # Loads .env variables
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def basic_chat_completion(user_prompt):

    system_prompt = f"""
    You are a psycho-analyst. You should be create meaningful summrization of the character and its relation to others.
    Given the following assistant responses about a literary character, summarize the character in three key-value pairs: 
    1. Emotional State, 
    2. Behavior Patterns, 
    3. Internal Conflicts. 
    Respond in JSON format like the following (DO NOT INCLUDE ANYTHING ELSE):
    {{
     "Emotional State": "Your Analysis",
     "Behavior Patterns": "Your Analysis",
     "Internal Conflicts": "Your Analysis"
    }}
    """


    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
            max_tokens=500,
            temperature=0.7
    
    )
    return response.choices[0].message.content

# ------------------- Utility for LLM-based merging -------------------

def enhance_text(existing_text: str, new_text: str) -> str:
    """Merge existing and new summaries via LLM for non-redundant enhanced text."""
    print("[DEBUG] Running enhance_text() in prompteng.py")
    print(f"[DEBUG] Top 5 chars of existing_text: {existing_text[:5]}")
    print(f"[DEBUG] Top 5 chars of new_text: {new_text[:5]}")

    if not new_text.strip():
        return existing_text.strip()

    prompt = f"""
You are refining a psychoanalytic summary.

Current Summary:
{existing_text.strip()}

New Insight:
{new_text.strip()}

Integrate them into a concise, non-redundant summary. Preserve meaning but avoid repetition. Return only the improved version.
"""
    improved = basic_chat_completion(prompt).strip()
    print(f"[DEBUG] Top 5 chars of improved: {improved[:5]}")
    return improved

# ------------------- Main summarization function -------------------

def summarize_character_from_chat():
    print("[DEBUG] Running summarize_character_from_chat() in prompteng.py")
    files_dir = os.path.join(os.path.dirname(__file__), 'files')
    chat_history_path = os.path.join(files_dir, 'chat_history.json')
    if not os.path.exists(chat_history_path):
        print("No chat history found.")
        return None

    with open(chat_history_path, 'r', encoding='utf-8') as f:
        chat_history = json.load(f)

    # Gather all assistant responses
    all_responses = ' '.join([turn['assistant']['content'] for turn in chat_history])
    print(f"[DEBUG] Top 5 chars of all_responses: {all_responses[:100]}")

    summary_prompt = f"Assistant Responses:\n{all_responses}"
    summary_response = basic_chat_completion(summary_prompt)

    print("Character Summary (from LLM):")
    print(summary_response)

    # --- Clean response to ensure valid JSON ---
    cleaned_response = summary_response.strip()
    if cleaned_response.startswith("```"):
        cleaned_response = re.sub(r"^```(?:json)?\s*", "", cleaned_response)
        cleaned_response = re.sub(r"\s*```$", "", cleaned_response).strip()

    print(f"[DEBUG] Top 5 chars of cleaned_response: {cleaned_response[:5]}")

    try:
        summary = json.loads(cleaned_response)
    except Exception:
        print("[WARN] LLM output is not valid JSON. No updates applied.")
        summary = {}

    print(f"[DEBUG] Parsed summary object: {summary}")

    # Load existing analysis (if any)
    os.makedirs(files_dir, exist_ok=True)
    analysis_path = os.path.join(files_dir, 'analysis_dictionary.json')
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            existing_analysis = json.load(f)
    else:
        existing_analysis = {}

    allowed_keys = {"Emotional State", "Behavior Patterns", "Internal Conflicts"}
    merged_analysis = existing_analysis.copy()

    if isinstance(summary, dict):
        for k in allowed_keys:
            new_val = summary.get(k, "").strip()
            existing_val = existing_analysis.get(k, "").strip()
            if new_val:
                if existing_val:
                    merged_val = enhance_text(existing_val, new_val)
                    merged_analysis[k] = merged_val
                else:
                    merged_analysis[k] = new_val

    print(f"[DEBUG] Merged analysis object before save: {merged_analysis}")

    # Always save the merged_analysis at the end
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(merged_analysis, f, ensure_ascii=False, indent=2)

    print(f"[DEBUG] Saved merged analysis to {analysis_path}")
    print(f"[DEBUG] Top 5 chars of merged_analysis: {str(merged_analysis)[:5]}")
    return merged_analysis

