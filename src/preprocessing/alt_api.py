import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import time
import sys
import itertools


MODEL_NAME = 'gemini-2.5-flash-lite'
BATCH_SIZE = 25
BATCH_DELAY_SECONDS = 15


try:
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / '.env'
    load_dotenv(dotenv_path=env_path)
except Exception as e:
    print(f"Warning: Could not load .env file. Error: {e}")

try:
    api_keys = [
        key for name, key in os.environ.items() 
        if name.startswith("GOOGLE_API_KEY_") and key.strip()
    ]

    if not api_keys:
        raise ValueError("No environment variables starting with 'GOOGLE_API_KEY_' found.")
    

    key_cycler = itertools.cycle(enumerate(api_keys, 1))
    
    print(f"Successfully loaded {len(api_keys)} API key(s).")
    

    genai.configure(api_key=api_keys[0])
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Configured Gemini API with model '{MODEL_NAME}'.")
    
except Exception as e:
    print(f"Error configuring Gemini client: {e}")
    exit()

def format_time(seconds):
    """Formats seconds into a human-readable string (h/m/s)."""
    if seconds < 0: return "0s"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0: return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
    if minutes > 0: return f"{int(minutes)}m {int(secs)}s"
    return f"{round(seconds)}s"

def process_comments_in_batch(comments_batch, current_api_key):
    """Processes a single batch of comments using the Gemini API."""
    prompt = (
        "You are an expert text processor. Your task is to process a JSON array of user comments.\n\n"
        "For each comment, determine its `classification` and produce a `cleaned_text` version based on the following rules:\n\n"
        "**Classification Categories:**\n"
        "* `rom_hindi`: For comments in Romanized Hindi (Hinglish).\n"
        "* `english`: For comments primarily in English.\n"
        "* `other`: For all other languages, including Hindi in its native Devanagari script.\n\n"
        "**Cleaning Rules:**\n"
        "* **For ALL comments:** Remove emojis. Do not transliterate native scripts.\n"
        "* **ONLY if `classification` is 'english':** In addition to removing emojis, also correct spelling and remove stop words.\n\n"
        "Your response must be a single, valid JSON array of objects. Each object must have two keys: \"classification\" and \"cleaned_text\". "
        "Maintain the original order and include no extra text or explanations.\n\n"
        f"Comments to process:\n{json.dumps(comments_batch, indent=2)}"
    )
    
    try:
        genai.configure(api_key=current_api_key)
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        result = json.loads(response.text)
        
        if isinstance(result, list) and len(result) == len(comments_batch):
            return result
        else:
            print(f"\nWarning: Batch returned a mismatched result. Expected {len(comments_batch)} items, got {len(result) if isinstance(result, list) else 'non-list'}.")
            return None
    except Exception as e:
        print(f"\nAn error occurred during API call: {e}")
        return None


input_filepath = os.path.join('..', '..', 'data', 'raw', 'youtube_data.json')
output_filepath = os.path.join('..', '..', 'data', 'clean', 'API_cleaned_data_combined.json')


try:
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file {input_filepath} was not found.")
    exit()


all_cleaned_data = {
    "rom_hindi": [],
    "english": []
}
seen_comments = set()

print(f"Starting data cleaning for the full dataset...")
overall_start_time = time.time()

video_count = 0
for video_object in data:
    if not isinstance(video_object, dict) or "video_id" not in video_object:
        continue

    video_count += 1
    video_id = video_object["video_id"]
    print(f"\nProcessing video {video_count}: {video_id}")
    
    comments_to_process = video_object.get("comments", [])
    total_comments = len(comments_to_process)
    
    if not comments_to_process:
        print("  - No comments found for this video. Skipping.")
        continue

    final_results_for_video = []
    video_start_time = time.time()
    
    for i in range(0, total_comments, BATCH_SIZE):
        batch = comments_to_process[i:i + BATCH_SIZE]
        

        key_index, current_key = next(key_cycler)
        batch_results = process_comments_in_batch(batch, current_key)
        
        if batch_results:
            final_results_for_video.extend(batch_results)
        else:
            print(f"  - Batch {i//BATCH_SIZE + 1} failed. Skipping these {len(batch)} comments.")

        comments_processed = min(i + BATCH_SIZE, total_comments)
        elapsed_time = time.time() - video_start_time
        cps = comments_processed / elapsed_time if elapsed_time > 0 else 0
        eta_seconds = ((elapsed_time / comments_processed) * (total_comments - comments_processed)) if cps > 0 else 0
        

        progress_bar = (
            f"  - Progress: {comments_processed}/{total_comments} | "
            f"Key: {key_index}/{len(api_keys)} | "
            f"CPS: {cps:.2f} | "
            f"ETA: {format_time(eta_seconds)}    "
        )
        sys.stdout.write('\r' + progress_bar)
        sys.stdout.flush()

        if (i + BATCH_SIZE) < total_comments:
            time.sleep(BATCH_DELAY_SECONDS)

    print()


    comments_added_this_video = 0
    for result in final_results_for_video:
        category = result.get("classification")
        cleaned_text = result.get("cleaned_text", "").strip()
        
        if cleaned_text and category in all_cleaned_data and cleaned_text not in seen_comments:
            all_cleaned_data[category].append(cleaned_text)
            seen_comments.add(cleaned_text)
            comments_added_this_video += 1
    
    print(f"  - Added {comments_added_this_video} new unique comments from this video.")


total_duration = time.time() - overall_start_time
print(f"\nTotal time taken: {format_time(total_duration)}")

output_dir = os.path.dirname(output_filepath)
os.makedirs(output_dir, exist_ok=True)


with open(output_filepath, 'w', encoding='utf-8') as f:
    json.dump(all_cleaned_data, f, indent=4, ensure_ascii=False)

print(f"Successfully processed {video_count} video(s) and saved the combined result.")
print(f"Final counts: {len(all_cleaned_data['rom_hindi'])} rom_hindi, {len(all_cleaned_data['english'])} english.")