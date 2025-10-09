import json, torch, warnings, os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore", category=UserWarning)

# === Model paths ===
BART_MODEL = "ai4bharat/IndicBART"
TRANS_MODEL = "ai4bharat/indictrans2-en-indic-1B"  # Or use indictrans2-all if preferred

DATA_PATH = "../data/API_cleaned_data_full.json"
OUT_PATH = "../data/indicbart_cleaned_outputs.json"

# === Device selection ===
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"🔹 Using device: {device}")
print("🔹 Loading IndicBART and IndicTrans2...")

# === Load models (with trust_remote_code for IndicTrans2) ===
bart_tok = AutoTokenizer.from_pretrained(BART_MODEL, use_fast=False)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL).to(device)

trans_tok = AutoTokenizer.from_pretrained(TRANS_MODEL, trust_remote_code=True)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL, trust_remote_code=True).to(device)

print("✅ Models loaded successfully!\n")


# === Helpers ===
def generate_hindi_counterspeech(comment: str):
    prompt = f"{comment}\n\nइस टिप्पणी के लिए एक सकारात्मक और सम्मानजनक उत्तर लिखिए।"
    inputs = bart_tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        outs = bart_model.generate(**inputs, max_length=96, num_beams=3, early_stopping=True)
    text = bart_tok.decode(outs[0], skip_special_tokens=True)
    return text.strip()

def translate_to_english(text: str):
    prefix = "translate Indic to English: " + text
    inputs = trans_tok(prefix, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.inference_mode():
        outs = trans_model.generate(**inputs, max_new_tokens=128)
    return trans_tok.decode(outs[0], skip_special_tokens=True).strip()

# === Load dataset ===
print(f"📖 Loading dataset from {DATA_PATH}...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)
entries = data if isinstance(data, list) else [{"video_id": k, **v} for k, v in data.items()]
print(f"✅ Loaded {len(entries)} video entries.\n")

# === Process ===
results = []
for entry in tqdm(entries, desc="Processing IndicBART"):
    vid = entry.get("video_id", "unknown")
    for lang, comments in entry.items():
        if lang in ["video_id", "model"]:
            continue
        if not isinstance(comments, list):
            continue
        for comment in comments:
            hindi_reply = generate_hindi_counterspeech(comment)
            english_reply = translate_to_english(hindi_reply)
            results.append({
                "video_id": vid,
                "lang": lang,
                "original_comment": comment,
                "counterspeech": english_reply,
                "model": "indicbart"
            })

# === Save ===
with open(OUT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n✅ Saved {len(results)} clean IndicBART responses to {OUT_PATH}")
