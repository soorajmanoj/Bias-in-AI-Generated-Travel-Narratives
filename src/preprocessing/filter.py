import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import time
import ijson

load_dotenv()

"""
@file filter.py
@brief Stream and classify YouTube comments using the Gemini API and write results to JSONL files.

Module-level configuration values control the input file, output files, rate limiting,
and the model instruction sent to the Gemini API. Functions in this module perform
single-comment classification, efficient appends to JSONL files, streaming processing
of comments for specified language keys, and an entry-point `main()` to run the
end-to-end pipeline.

@note Requires environment variable `GOOGLE_API_KEY_8` to be set and valid.
"""

API_KEY = os.getenv("GOOGLE_API_KEY_8")
if API_KEY:
    print("API Key loaded successfully.")
else:
    print("Error: API_KEY not found in environment variables.")
genai.configure(api_key=API_KEY)

INPUT_FILE = "../../data/clean/final_API_data.json"
RELEVANT_OUTPUT_FILE = "../../data/clean/filtered/relevant.jsonl"
IRRELEVANT_OUTPUT_FILE = "../../data/clean/filtered/irrelevant.jsonl"
ERROR_OUTPUT_FILE = "../../data/clean/filtered/error.jsonl"

RATE_LIMIT_DELAY = 0.2

MODEL_INSTRUCTION = """
You are analyzing YouTube comments on travel vlogs related to India for a project on societal biases.
Your task is to classify each comment as “relevant” or “irrelevant” based on a strict definition.

Definition of Relevance:
A comment is relevant *only if* it expresses a clear opinion, generalization, or comparison about:
* Broad Societal Topics: Indian culture, society, safety, politics, religion, traditions, or lifestyle (e.g., "India is so unsafe for women," "Indian culture is very spiritual").
* Explicit Judgments: Direct praise, criticism, or controversy about India, its people, or its culture as a whole (e.g., "Indians are the friendliest people," "India is a very dirty country").
* Direct Comparisons: Explicit comparisons between India and other countries (e.g., "It's cleaner here than in Pakistan," "People in Europe are not as welcoming as in India").
* Generalizations from Travel: Travel experiences that are used to make a broader conclusion about the country or its people (e.g., "I got scammed, this happens all the time in India").

Definition of Irrelevance:
A comment is irrelevant if it:
* Is a Personal Anecdote: Describes a simple, personal interaction or a specific event *without* making a broader judgment (e.g., "The lady on the street gave me an apple," "Our guide was very nice").
* Is a Simple Observation: Makes a neutral observation about food, prices, or scenery (e.g., "That food looks delicious," "The mountains are beautiful," "The train was late").
* Focuses on the Creator: Talks about the vlogger, their music, editing, or unrelated topics (e.g., "Love your videos!", "What camera do you use?").
* Is Generic: Contains only emojis, tags, links, timestamps, spam, promotions, or random text.

Output format:
Return results as JSON with two fields:

{
  "comment": "<original comment>",
  "classification": "relevant" or "irrelevant"
}
"""

model = genai.GenerativeModel('gemini-2.5-flash-lite')
generation_config = genai.GenerationConfig(response_mime_type="application/json")


def classify_comment(comment_text):
    """
    @brief Classify a single comment via the Gemini API.

    @param comment_text The comment text to classify.
    @return A lowercase classification string: "relevant", "irrelevant", or "error".
    """
    prompt_for_api = f"{MODEL_INSTRUCTION}\n\n---\nPlease classify the following comment:\n\"{comment_text}\""

    try:
        response = model.generate_content(
            prompt_for_api,
            generation_config=generation_config
        )
        result_data = json.loads(response.text)

        classification_dict = None

        if isinstance(result_data, list):
            if len(result_data) > 0:
                classification_dict = result_data[0]
            else:
                print(f"--- API returned an empty list for comment: {comment_text[:50]}...")
                return "ERROR"

        elif isinstance(result_data, dict):
            classification_dict = result_data

        else:
            print(f"--- API returned unexpected format for comment: {comment_text[:50]}...")
            return "ERROR"

        return classification_dict.get("classification", "ERROR").lower()

    except Exception as e:
        print(f"--- ERROR classifying comment: {comment_text[:50]}...")
        print(f"--- Error details: {e}")
        return "ERROR"


def append_to_jsonl(filename, comment, language):
    """
    @brief Append a JSON object as a single line to a .jsonl file.

    @param filename Path to the JSONL file to append to.
    @param comment The comment text to include in the object.
    @param language The language key associated with the comment.
    @return None
    """
    output_data = {
        "comment": comment,
        "language": language
    }
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

    except IOError as e:
        print(f"--- ERROR writing to {filename}: {e}")


def process_comments(language_key):
    """
    @brief Stream comments for a language key from the configured input file, classify each
    comment, and write to the appropriate output JSONL files.

    @param language_key The key within the input JSON whose items will be processed.
    @return None
    """
    print(f"\n--- Starting processing for: {language_key} ---")
    total_count = 0
    relevant_count = 0
    irrelevant_count = 0
    error_count = 0

    try:
        with open(INPUT_FILE, 'rb') as f:
            comments_stream = ijson.items(f, f'{language_key}.item')

            for comment in comments_stream:
                total_count += 1

                if not isinstance(comment, str) or not comment.strip():
                    print(f"Skipping empty/invalid comment #{total_count}")
                    append_to_jsonl(IRRELEVANT_OUTPUT_FILE, str(comment), language_key)
                    irrelevant_count += 1
                    continue

                print(f"Processing {language_key} comment {total_count}: {comment[:60]}...")

                classification = classify_comment(comment)

                if classification == "relevant":
                    append_to_jsonl(RELEVANT_OUTPUT_FILE, comment, language_key)
                    relevant_count += 1
                elif classification == "irrelevant":
                    append_to_jsonl(IRRELEVANT_OUTPUT_FILE, comment, language_key)
                    irrelevant_count += 1
                else:
                    append_to_jsonl(ERROR_OUTPUT_FILE, comment, language_key)
                    error_count += 1

                time.sleep(RATE_LIMIT_DELAY)

    except ijson.JSONError as e:
        print(f"Error parsing JSON from {INPUT_FILE}. Is it valid? Error: {e}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")

    print(f"\n--- Finished processing for: {language_key} ---")
    print(f"Total: {total_count}, Relevant: {relevant_count}, Irrelevant: {irrelevant_count}, Errors: {error_count}")


def main():
    """
    @brief Entry point to clear output files and process the configured language keys.

    @return None
    """
    open(RELEVANT_OUTPUT_FILE, 'w').close()
    open(IRRELEVANT_OUTPUT_FILE, 'w').close()
    open(ERROR_OUTPUT_FILE, 'w').close()
    print("Cleared all output files.")

    process_comments('rom_hindi')
    process_comments('english')

    print("\n--- All processing complete! ---")


if __name__ == "__main__":
    main()