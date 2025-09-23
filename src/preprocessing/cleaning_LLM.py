import os
import json
import ollama


def process_comment_with_local_llm(comment, model_name='gemma3:1b'):
    """
    Sends a comment to a local Ollama model for cleaning and classification.
    """
    
    system_prompt = (
        "You are a text processing expert. For the user comment, perform these tasks: "
        "1. Clean the text by correcting spelling and grammar mistakes and removing all emojis. Do not remove stop words. "
        "2. Classify the comment's primary language into one of three categories: 'rom_hindi', 'english', or 'other'. "
        "Provide your response only as a valid JSON object with two keys: 'classification' and 'cleaned_text'."
    )
    
    try:
       
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': comment},
            ],
            format='json' 
        )
        
       
        result = json.loads(response['message']['content'])
        
        if "classification" in result and "cleaned_text" in result:
            return result
        else:
            return None

    except Exception as e:
        print(f"An error occurred while calling the local model: {e}")
        return None


input_filepath = os.path.join('..', '..', 'data', 'raw', 'youtube_data.json')
output_filepath = os.path.join('..', '..', 'data', 'clean', 'local_llm_cleaned_data.json')


try:
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: The file {input_filepath} was not found.")
    exit()

all_cleaned_data = []
print("🚀 Starting data cleaning process with local LLM...")

for video_object in data:
    if not isinstance(video_object, dict) or "video_id" not in video_object:
        continue

    cleaned_data = {
        "video_id": video_object["video_id"],
        "rom_hindi": [],
        "english": [],
        "other": []
    }
    
    for comment in video_object.get("comments", []):
        if not isinstance(comment, str) or not comment.strip():
            continue
        
        print(f"Processing comment: '{comment[:40]}...'")
        
        llm_result = process_comment_with_local_llm(comment)
        
        if llm_result:
            category = llm_result.get("classification", "other")
            cleaned_text = llm_result.get("cleaned_text", "")
            
            if category in cleaned_data:
                cleaned_data[category].append(cleaned_text)
            else:
                cleaned_data["other"].append(cleaned_text)

    all_cleaned_data.append(cleaned_data)


output_dir = os.path.dirname(output_filepath)
os.makedirs(output_dir, exist_ok=True)
with open(output_filepath, 'w', encoding='utf-8') as f:
    json.dump(all_cleaned_data, f, indent=4, ensure_ascii=False)

print(f"✅ Successfully processed {len(all_cleaned_data)} video(s) and saved the result to {output_filepath}")