import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------
# LOAD MODEL (Qwen 2.5 - 3B Instruct on MPS)
# ----------------------------------------
print("🔹 Loading Qwen2.5-3B-Instruct on mps...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map=device,
    trust_remote_code=True
)

print("✨ Model loaded successfully!")


# ----------------------------------------
# LOAD MERGED JSON (same structure as before)
# ----------------------------------------
INPUT_FILE = "../data/API_cleaned_data_full.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

rom_hindi_comments = data.get("rom_hindi", [])
english_comments = data.get("english", [])

print(f"📌 Loaded {len(rom_hindi_comments)} Roman Hindi comments")
print(f"📌 Loaded {len(english_comments)} English comments")


# ----------------------------------------
# COUNTERSPEECH PROMPT (ENGLISH ONLY)
# ----------------------------------------
SYSTEM_PROMPT = """
You are a counterspeech assistant.

Always respond in **English only**, even if the user's comment is in
Romanized Hindi or Hinglish.

Guidelines:
- Keep replies short (1–2 sentences)
- Simple, natural English
"""

def build_prompt(comment):
    return f"{SYSTEM_PROMPT}\nUser: {comment}\nAssistant:"


# ----------------------------------------
# GENERATION FUNCTION
# ----------------------------------------
def generate_counterspeech(comment: str) -> str:
    prompt = build_prompt(comment)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean prompt echo
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    return text


# ----------------------------------------
# PROCESS ALL COMMENTS (same as llama_client)
# ----------------------------------------
output = []

all_comments = (
    [(c, "rom_hindi") for c in rom_hindi_comments] +
    [(c, "english") for c in english_comments]
)

print(f"📝 Generating counterspeech for {len(all_comments)} total comments...")

for comment, lang in all_comments:
    reply = generate_counterspeech(comment)
    output.append({
        "comment": comment,
        "language": lang,
        "counterspeech_english": reply
    })


# ----------------------------------------
# SAVE OUTPUT
# ----------------------------------------
OUTPUT_FILE = "../outputs/qwen25_counterspeech_output.json"

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Finished!")
print(f"💾 Saved to {OUTPUT_FILE}")
