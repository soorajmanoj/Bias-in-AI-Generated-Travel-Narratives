import json
import time
import requests
import os
import re
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")

if not PERSPECTIVE_API_KEY:
    raise ValueError("PERSPECTIVE_API_KEY not found in .env")

API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

ATTRS = {
    "TOXICITY": {},
    "SEVERE_TOXICITY": {},
    "INSULT": {},
    "PROFANITY": {},
    "IDENTITY_ATTACK": {}
}

CHUNK_SAVE_INTERVAL = 25  # save progress every 25 comments


def clean(text):
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[\x00-\x1F\x7F]", "", text).strip()


def score_comment(comment):
    """Safe Perspective API call with retry and backoff."""
    payload = {
        "comment": {"text": clean(comment)},
        "languages": ["en"],
        "requestedAttributes": ATTRS,
    }

    while True:
        resp = requests.post(API_URL, json=payload, params={"key": PERSPECTIVE_API_KEY})

        if resp.status_code == 200:
            data = resp.json()
            scores = {}
            for a in ATTRS:
                try:
                    scores[a] = data["attributeScores"][a]["summaryScore"]["value"]
                except KeyError:
                    scores[a] = None
            return scores

        elif resp.status_code == 429:
            print(" Rate limit exceeded — waiting 10 seconds...")
            time.sleep(10)
            continue  # retry

        else:
            print(f" API ERROR {resp.status_code}: {resp.text}")
            return {a: None for a in ATTRS}


def main():
    input_file = "../counterspeech/outputs/llama_partial.json"
    output_file = "../counterspeech/outputs/llama32_perspective_scores_final.json"

    print(f" Loading input: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    start_index = 0

    # Resume support
    if os.path.exists(output_file):
        print(" Resuming from existing output...")
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_index = len(results)
        print(f"➡ Starting from index {start_index}/{len(data)}")

    print(" Scoring comments...\n")

    for i in tqdm(range(start_index, len(data))):
        comment = data[i]["counterspeech_english"]
        language = data[i]["language"]
        score = score_comment(comment)

        results.append({"comment": comment, "lang": language, "perspective_scores": score})

        # Save checkpoint every 25 items
        if i % CHUNK_SAVE_INTERVAL == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        # SAFE RATE LIMIT: 2 seconds
        time.sleep(1.5)

    print("\n Final save...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(" Done")


if __name__ == "__main__":
    main()
