import google.generativeai as genai
import json
import os
import time

# --- 1. Configuration ---

# PLEASE SET YOUR API KEY
# Option 1 (Recommended): Set an environment variable named 'GOOGLE_API_KEY'
# in your terminal: export GOOGLE_API_KEY='Your_Key_Here' (macOS/Linux)
#                  set GOOGLE_API_KEY='Your_Key_Here' (Windows)
API_KEY = os.getenv("GOOGLE_API_KEY")

# Option 2: Paste your key here (less secure)
if not API_KEY:
    API_KEY = "AIzaSyATP22cZ1ThXI2A71AHyMs3Svj6XYOLCiQ"

# if not API_KEY:
#     raise ValueError("Please set the GOOGLE_API_KEY environment variable or paste your key directly into the script.")

genai.configure(api_key=API_KEY)

# --- 2. File Configuration ---
INPUT_FILE = "test/data.json"  # <-- Change this to your file's name if different
OUTPUT_FILE = "test/classified_comments.json"

# Delay between API calls (in seconds) to avoid rate limiting
RATE_LIMIT_DELAY = 1

# --- 3. Model Instruction (Copied directly from your prompt) ---
MODEL_INSTRUCTION = """
You are analyzing YouTube comments on travel vlogs related to India.
Your task is to classify each comment as “relevant” or “irrelevant.”

Definition of relevance:
A comment is relevant if it discusses, mentions, or implies anything about:
	•	India or other countries, especially in comparison.
	•	Culture, society, safety, politics, food, people, tourism, religion, lifestyle, or traditions.
	•	Any opinions, praise, criticism, or controversy about India or Indian culture.
	•	Travel experiences, infrastructure, or local interactions in India.

A comment is irrelevant if it:
	•	Only contains emojis, tags, links, timestamps, or random text.
	•	Talks about the creator, music, editing, or unrelated topics.
	•	Is spam, promotional, or completely off-topic.

Output format:
Return results as JSON with two fields:

{
  "comment": "<original comment>",
  "classification": "relevant" or "irrelevant"
}
"""

# --- 4. Setup Gemini Model ---
# We configure the model to *only* output JSON.
model = genai.GenerativeModel('gemini-2.5-flash-lite')
generation_config = genai.GenerationConfig(response_mime_type="application/json")


# --- 5. Helper Function for Classification ---
def classify_comment(comment_text):
    """
    Sends a single comment to the Gemini API for classification.
    """
    # Create the full prompt to send to the API
    prompt_for_api = f"""{MODEL_INSTRUCTION}

---
Please classify the following comment:
"{comment_text}"
"""

    try:
        response = model.generate_content(
            prompt_for_api,
            generation_config=generation_config
        )
        # The response text is a JSON string, so we parse it into a Python dict
        result = json.loads(response.text)
        return result

    except Exception as e:
        print(f"--- ERROR classifying comment: {comment_text[:50]}...")
        print(f"--- Error details: {e}")
        # Return a consistent error format if the API fails
        return {
            "comment": comment_text,
            "classification": "ERROR"
        }


# --- 6. Main Script Logic ---
def main():
    print(f"Loading comments from {INPUT_FILE}...")

    # Load the input JSON file
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {INPUT_FILE}. Please check the file format.")
        return

    # Combine all comments into a single list
    rom_hindi_comments = data.get("rom_hindi", [])
    english_comments = data.get("english", [])
    all_comments = rom_hindi_comments + english_comments

    if not all_comments:
        print("No comments found in the input file.")
        return

    print(f"Found {len(all_comments)} total comments to process.")

    all_results = []

    # Process each comment one by one
    for i, comment in enumerate(all_comments):
        print(f"Processing comment {i + 1}/{len(all_comments)}: {comment[:60]}...")

        # Skip empty or invalid comments
        if not isinstance(comment, str) or not comment.strip():
            print("Skipping empty or invalid comment.")
            all_results.append({
                "comment": str(comment),
                "classification": "irrelevant"  # Treat empty/invalid as irrelevant
            })
            continue

        result_data = classify_comment(comment)
        all_results.append(result_data)

        # Be nice to the API and respect rate limits
        time.sleep(RATE_LIMIT_DELAY)

    # Save all results to the output file
    print(f"\nProcessing complete. Saving results to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # indent=2 makes the JSON human-readable
            # ensure_ascii=False correctly saves non-English characters
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print("Successfully saved results.")
    except IOError as e:
        print(f"Error saving output file: {e}")


# Run the script
if __name__ == "__main__":
    main()