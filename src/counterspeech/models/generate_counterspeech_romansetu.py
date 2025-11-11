import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# ============================================================
# 1. Model Setup
# ============================================================

MODEL_PATH = (
    "/Users/akken/Desktop/Projects_nodel/"
    "Bias-in-AI-Generated-Travel-Narratives/src/"
    "counterspeech/models/models_cache/romansetu-cpt-roman-sft-roman"
)

print(f"🔹 Loading local model from: {MODEL_PATH}")

# --- Auto-detect device ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"🧠 Using device: {device.upper()}")

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# half precision for MPS; else float32
dtype = torch.float16 if device == "mps" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map=None
)
model.to(device)
model.eval()

print(f"✅ Model loaded successfully on {device.upper()}.\n")

# ============================================================
# 2. Generation Function
# ============================================================

def generate_counterspeech(comment: str) -> str:
    """Generate a kind, respectful counterspeech reply for a given comment."""
    prompt = (
        f"Generate a kind and respectful counterspeech reply for the following comment:\n"
        f"{comment}\n\nReply:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,  # Greedy decoding
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Reply:" in text:
        text = text.split("Reply:")[-1].strip()
    return text

# ============================================================
# 3. Load Dataset
# ============================================================

INPUT_FILE = "../data/API_cleaned_data_full.json"
OUTPUT_FILE = "counterspeech_output.json"

print(f"📂 Loading dataset from {INPUT_FILE}")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)[0]

results = {"english": [], "rom_hindi": []}

# ============================================================
# 4. Process Comments
# ============================================================

def process_comments(lang_key: str):
    comments = data[lang_key]
    print(f"\n🗣️ Generating counterspeech for {lang_key} comments ({len(comments)} total)...")

    for i, comment in enumerate(tqdm(comments, desc=f"{lang_key}")):
        try:
            print(f"🕒 Generating for: {comment[:60]}...")
            counterspeech = generate_counterspeech(comment)
            print(f"✅ Done → {counterspeech[:80]}...")
        except Exception as e:
            print(f"⚠️ Error generating for comment {i}: {e}")
            counterspeech = "Generation failed."

        results[lang_key].append({
            "comment": comment,
            "counterspeech": counterspeech
        })

        # Intermediate checkpoint every 5 comments
        if (i + 1) % 5 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"💾 Progress saved after {i+1} {lang_key} comments.")

# Run both sets
process_comments("english")
process_comments("rom_hindi")

# ============================================================
# 5. Save Final Results
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✅ All counterspeech replies generated and saved to {OUTPUT_FILE}")
