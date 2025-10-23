import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# ---------- CONFIG ----------
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
INPUT_FILE = "../data/API_cleaned_data_full.json"
OUTPUT_FILE = "../outputs/counterspeech_output_llama32.json"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ---------- LOAD MODEL ----------
print(f"🔹 Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
print(" Model loaded successfully!")

# ---------- LOAD INPUT DATA ----------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

english_comments = data[0].get("english", [])
rom_hindi_comments = data[0].get("rom_hindi", [])

# ---------- DEFINE PROMPT ----------
def build_prompt(comment, lang):
    if lang == "english":
        return f"""You are a youtube assistant
Respond to the following English comment with a relevant reply.

Comment: "{comment}"
Counterspeech:"""
    else:
        return f"""You are a youtube assistant.
The following comment is written in Romanized Hindi. Respond in **English** with a relevant reply.
Comment (Romanized Hindi): "{comment}"
Counterspeech:"""

# ---------- GENERATION FUNCTION ----------
def generate_counterspeech(comment, lang):
    prompt = build_prompt(comment, lang)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract model output text after "Counterspeech:"
    if "Counterspeech:" in response:
        return response.split("Counterspeech:")[-1].strip()
    return response.strip()

# ---------- PROCESS ----------
output_data = {"english": [], "rom_hindi": []}

print(" Generating counterspeech for English comments...")
for comment in tqdm(english_comments, desc="English"):
    try:
        counterspeech = generate_counterspeech(comment, "english")
        output_data["english"].append({"comment": comment, "counterspeech": counterspeech})
    except Exception as e:
        print(f" Error on English comment: {e}")

print(" Generating counterspeech for Romanized Hindi comments (responses in English)...")
for comment in tqdm(rom_hindi_comments, desc="Romanized Hindi"):
    try:
        counterspeech = generate_counterspeech(comment, "rom_hindi")
        output_data["rom_hindi"].append({"comment": comment, "counterspeech": counterspeech})
    except Exception as e:
        print(f" Error on Romanized Hindi comment: {e}")

# ---------- SAVE OUTPUT ----------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f" Counterspeech generation complete! Saved to {OUTPUT_FILE}")
