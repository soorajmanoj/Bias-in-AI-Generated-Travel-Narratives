import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------
# LOAD MODEL (Llama-3.2-1B-Instruct, MPS)
# ----------------------------------------
print("🔹 Loading meta-llama/Llama-3.2-1B-Instruct on mps...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map=device,
    trust_remote_code=True
)

print("✨ Model loaded successfully!")


# ----------------------------------------
# LOAD YOUR MERGED JSON FILE
# ----------------------------------------
INPUT_FILE = "../data/API_cleaned_data_full2.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

rom_hindi_comments = data.get("rom_hindi", [])
english_comments = data.get("english", [])

print(f"📌 Loaded {len(rom_hindi_comments)} Roman Hindi comments")
print(f"📌 Loaded {len(english_comments)} English comments")


# ----------------------------------------
# COUNTERSPEECH PROMPT (ENGLISH ONLY)
# ----------------------------------------
# SYSTEM_PROMPT = """
# You are a counterspeech assistant.
#
# Your job is to reply **only in English**, even when the input comment
# is in Romanized Hindi or Hinglish.
#
# Guidelines:
# - 1–2 sentence reply
# - Natural, simple English
# - Avoid jargon or complex vocabulary
# """
SYSTEM_PROMPT = """
Reply in English only.

You are allowed to be:
- sarcastic
- rude
- blunt and brutally honest
- unimpressed or dismissive
Guidelines:
- Use 1–2 sentences.
Keep replies short and sharp.
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt echo if present
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    return text


# ----------------------------------------
# PROCESS ALL COMMENTS
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
OUTPUT_FILE = "../outputs/llama32_counterspeech_output_final.json"

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Finished!")
print(f"💾 Saved to {OUTPUT_FILE}")
