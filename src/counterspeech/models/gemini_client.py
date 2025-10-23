import os, json, google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# === Setup ===
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DATA_PATH = "../data/API_cleaned_data_full.json"
OUT_PATH = "../data/gemini_responses.json"
model = genai.GenerativeModel("gemini-1.5-flash")

def build_prompt(comment: str) -> str:
    return f"""You are a respectful counterspeech assistant.

Generate a polite, educational counterspeech message in English
for the following YouTube comment:

"{comment}"

Output format (strict JSON):
{{"original_comment": "{comment}", "counterspeech": "your response here"}}
"""

def get_gemini_response(comment: str):
    prompt = build_prompt(comment)
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
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
for vid, langs in tqdm(data.items(), desc="Processing Gemini"):
    for lang, comments in langs.items():
        for comment in comments:
            res = get_gemini_response(comment)
            res.update({"video_id": vid, "lang": lang, "model": "gemini"})
            results.append(res)

with open(OUT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f" Saved {len(results)} Gemini responses to {OUT_PATH}")
