import json
import os
from tqdm import tqdm
from detoxify import Detoxify

"""
@file detoxify_scores.py
@brief Score counterspeech text locally using the Detoxify Multilingual model.
"""

# Load the multilingual model to support both English and Romanized Hindi
print("Loading Detoxify Multilingual model...")
detox_model = Detoxify('multilingual')
print("Model loaded.")

CHUNK_SAVE_INTERVAL = 25


def score_comment(comment):
    """
    @brief Local scoring using Detoxify. No rate limits, no API keys.
    """
    if not comment or not isinstance(comment, str):
        return {}

    try:
        # Detoxify returns a dictionary of scores
        scores = detox_model.predict(comment)

        # Convert NumPy float32 types to standard Python floats so JSON can save them
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        print(f" Scoring Error: {e}")
        return {}


def main():
    input_file = "../../data/clean/filtered/qwen25_counterspeech_output_final.json"
    output_file = "../counterspeech/outputs/qwen25_detoxify_scores_final.json"

    print(f" Loading input: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    start_index = 0

    # Checkpointing logic: Resume if partial file exists
    if os.path.exists(output_file):
        print(" Resuming from existing output...")
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_index = len(results)
        print(f"➡ Starting from index {start_index}/{len(data)}")

    print(" Scoring comments locally...\n")

    for i in tqdm(range(start_index, len(data))):
        comment = data[i]["counterspeech_english"]
        language = data[i]["language"]

        score = score_comment(comment)

        # Append to results list
        results.append({"comment": comment, "lang": language, "perspective_scores": score})

        # Save progress every CHUNK_SAVE_INTERVAL items
        if i > 0 and i % CHUNK_SAVE_INTERVAL == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save when complete
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(" Done")


if __name__ == "__main__":
    main()