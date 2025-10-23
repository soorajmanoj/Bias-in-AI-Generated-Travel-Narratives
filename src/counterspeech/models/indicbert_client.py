import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json, os

# =======================
# Device setup
# =======================
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"🔹 Using device: {device}")

# =======================
# Load Sarvam-2B
# =======================
MODEL_ID = "sarvamai/Sarvam-2B"
print(f"🔹 Loading model {MODEL_ID} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device != "mps" else torch.float32,
).to(device)

print("✅ Model loaded successfully!\n")

# =======================
# Prompt builder
# =======================
def build_prompt(comment: str):
    return (
        "You are a polite and culturally aware Indian assistant. "
        "Read the following comment written in English or Romanized Hindi. "
        "Generate counterspeech for.\n\n"
        f"Comment: {comment}\nCounterspeech:"
    )

# =======================
# Generation function
# =======================
@torch.no_grad()
def generate_counterspeech(comment: str) -> str:
    prompt = build_prompt(comment)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Counterspeech:" in text:
        text = text.split("Counterspeech:")[-1].strip()
    return text

# =======================
# Main pipeline
# =======================
if __name__ == "__main__":
    data_path = "../data/API_cleaned_data_full.json"
    output_path = "../../outputs/sarvam_counterspeech_results.json"

    print(f"📖 Loading dataset from {data_path}...")
    with open(data_path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} video entries.\n")

    results = []
    for entry in tqdm(data, desc="Processing Videos"):
        video_id = entry.get("video_id", "unknown")
        comments = []
        for field in ["rom_hindi", "english", "other"]:
            comments.extend(entry.get(field, []))

        for i, comment in enumerate(comments):
            if not comment.strip():
                continue
            try:
                counterspeech = generate_counterspeech(comment)
                results.append({
                    "video_id": video_id,
                    "comment_id": f"{video_id}_{i+1}",
                    "original_comment": comment,
                    "counterspeech": counterspeech
                })
            except Exception as e:
                print(f"❌ Error processing comment {i+1} in {video_id}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. {len(results)} counterspeech responses saved to {output_path}")
