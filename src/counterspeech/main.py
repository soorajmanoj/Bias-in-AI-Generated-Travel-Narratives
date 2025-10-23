import json, os
from tqdm import tqdm
from models.llama_client import get_gpt4_response
from models.gemini_client import get_gemini_response
from models.indicbert_client import get_indic_response

DATA_PATH = "data/API_cleaned_data_full.json"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Suppress TensorFlow / gRPC noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"


def safe_load_json(path):
    """Safely load dataset, converting list-of-dicts into dict by video_id."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        elif isinstance(data, list):
            normalized = {}
            for i, entry in enumerate(data):
                vid = entry.get("video_id") or f"vid_{i+1}"
                normalized[vid] = {
                    "english": entry.get("english", []),
                    "romanized_hindi": entry.get("romanized_hindi", []),
                    "other": entry.get("other", [])
                }
            print(f"✅ Converted list of {len(normalized)} entries into dict by video_id.")
            return normalized

        else:
            raise ValueError("Unsupported JSON format (must be dict or list).")

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load dataset {path}: {e}")


def process_and_save(model_func, model_name, data, output_path):
    """Run the selected model across dataset and save its own JSON output."""
    results = []
    print(f"\n🚀 Running {model_name} on dataset...")

    for video_id, comments_by_lang in tqdm(data.items(), desc=f"{model_name}"):
        for lang, comments in comments_by_lang.items():
            for comment in comments:
                try:
                    response = model_func(comment)
                    results.append({
                        "video_id": video_id,
                        "language": lang,
                        "input": comment,
                        "output": response
                    })
                except Exception as e:
                    results.append({
                        "video_id": video_id,
                        "language": lang,
                        "input": comment,
                        "error": str(e)
                    })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {model_name} results to {output_path} ({len(results)} entries).")


def main():
    data = safe_load_json(DATA_PATH)
    print(f"✅ Loaded dataset with {len(data)} videos.")

    process_and_save(get_gpt4_response, "GPT-4", data, os.path.join(OUTPUT_DIR, "gpt4_responses.json"))
    process_and_save(get_gemini_response, "Gemini", data, os.path.join(OUTPUT_DIR, "gemini_responses.json"))
    process_and_save(lambda c: get_indic_response(c, target_lang="en"), "IndicInstruct", data, os.path.join(OUTPUT_DIR, "indicbert_responses.json"))

if __name__ == "__main__":
    main()
