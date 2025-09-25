import os
import json
import ollama
import re
import time
import sys


MODEL_NAME = 'deepseek-r1:8b'
COMMENT_LIMIT = 100 

def format_time(seconds):
    """Converts seconds into a human-readable H:M:S format."""
    if seconds < 0:
        return "0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def extract_json_from_string(text):
    """Finds and extracts the first valid JSON object from a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return None
    return None

def process_comment_with_llm(comment, model_name=MODEL_NAME):
    """Sends a single comment to the LLM for cleaning and classification."""
    system_prompt = (
        "You are a text processing expert. For the user comment, perform these tasks: "
        "dont change the comment but just classify the language into 'rom_hindi'(romanized hindi), 'english', or 'other. "
        "Provide your response only as a valid JSON object with two keys: 'classification' and 'cleaned_text'."
    )
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': comment}],
            format='json'
        )
        raw_response_text = response['message']['content']
        return extract_json_from_string(raw_response_text)
    except Exception as e:
        print(f"\n  - Error processing comment: {e}")
        return None


input_filepath = os.path.join('..', '..', 'data', 'raw', 'youtube_data.json')
output_filepath = os.path.join('..', '..', 'data', 'clean','tests', 'llm_cleaned_data_test.json')

try:
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: The file {input_filepath} was not found.")
    exit()

all_cleaned_data = []
print(f"🚀 Starting data cleaning process with model '{MODEL_NAME}' (limit: {COMMENT_LIMIT} comments)...")

for video_object in data:
    if not isinstance(video_object, dict) or "video_id" not in video_object:
        continue

    video_id = video_object["video_id"]
    print(f"Processing video: {video_id}")
    

    comments_to_process = video_object.get("comments", [])[:COMMENT_LIMIT]
    total_comments = len(comments_to_process)
    
    if not comments_to_process:
        break 

    cleaned_data = {
        "video_id": video_id,
        "rom_hindi": [],
        "english": [],
        "other": []
    }

    start_time = time.time()
    for i, comment in enumerate(comments_to_process):
        if not isinstance(comment, str) or not comment.strip():
            continue
        
        result = process_comment_with_llm(comment)
        
        if result:
            category = result.get("classification", "other")
            cleaned_text = result.get("cleaned_text", "")
            if category in cleaned_data:
                cleaned_data[category].append(cleaned_text)
            else:
                cleaned_data["other"].append(cleaned_text)
        
        if i > 0:
            elapsed_time = time.time() - start_time

            cps = (i + 1) / elapsed_time 
            comments_remaining = total_comments - (i + 1)
            eta_seconds = (elapsed_time / (i + 1)) * comments_remaining
            formatted_eta = format_time(eta_seconds)
            
            progress_bar = f"  - Processing comment {i + 1}/{total_comments}... CPS: {cps:.2f} | ETA: {formatted_eta}   "
            sys.stdout.write('\r' + progress_bar)
            sys.stdout.flush()

    print() 
    all_cleaned_data.append(cleaned_data)

    break

output_dir = os.path.dirname(output_filepath)
os.makedirs(output_dir, exist_ok=True)
with open(output_filepath, 'w', encoding='utf-8') as f:
    json.dump(all_cleaned_data, f, indent=4, ensure_ascii=False)

print(f"✅ Successfully processed {len(all_cleaned_data)} video(s) and saved the result.")