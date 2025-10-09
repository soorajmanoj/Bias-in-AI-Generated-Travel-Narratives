import os, json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# === Setup ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = "src/counterspeech/data/API_cleaned_data_full.json"
OUT_PATH = "src/counterspeech/data/gpt4_responses.json"

def build_prompt(comment: str) -> str:
    return f"""You are a respectful counterspeech assistant.

Generate a polite, educational counterspeech message in English
for the following YouTube comment:

"{comment}"

Output format (strict JSON):
{{"original_comment": "{comment}", "counterspeech": "your response here"}}
"""

def get_gpt4_response(comment: str):
    prompt = build_prompt(comment)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except Exception as e:
        return {"original_comment": comment, "counterspeech": f"Error: {e}"}
    return {"original_comment": comment, "counterspeech": text}

# === Execution ===
with open(DATA_PATH, "r") as f:
    data = json.load(f)

results = []
for vid, langs in tqdm(data.items(), desc="Processing GPT-4"):
    for lang, comments in langs.items():
        for comment in comments:
            res = get_gpt4_response(comment)
            res.update({"video_id": vid, "lang": lang, "model": "gpt4"})
            results.append(res)

with open(OUT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(results)} GPT-4 responses to {OUT_PATH}")
