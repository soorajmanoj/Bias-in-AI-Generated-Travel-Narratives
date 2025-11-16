import json
import time
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

MODEL_NAME = "sarvamai/Airavata-LLM-SFT-v2-7B"

print("🔥 Loading Airavata 7B in 4-bit mode (English-only counterspeech)…")

# ---------------------------
# Quantization config (macOS)
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=device,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

SYSTEM_PROMPT = """
You are an Indian counterspeech assistant.

Your job is to write calm, respectful, and empathetic replies
ONLY in English, regardless of the language of the input comment.

Rules:
- Do NOT use Hindi, Hinglish, or Roman Hindi words in your reply.
- Keep replies short (1–2 sentences).
- Do not insult the user.
- Focus on reducing tension, promoting understanding, and being polite.
- Use simple English that feels natural and human.
"""

def build_prompt(comment):
    return f"{SYSTEM_PROMPT}\n\nUser comment: {comment}\nAssistant reply (English only):"

def generate_counterspeech(text: str) -> str:
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=70,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove echoes of the prompt if model repeats it
    if "Assistant reply" in reply:
        reply = reply.split("Assistant reply (English only):")[-1].strip()

    return reply

def main():
    input_file = "../data/API_cleaned_data_full.json"
    output_file = "../outputs/airavata_english_counterspeech.json"

    with open(input_file, "r") as f:
        records = json.load(f)

    all_results = []

    for record in records:
        video_id = record["video_id"]
        comments = record["rom_hindi"] + record["english"] + record["other"]

        for c in tqdm(comments, desc=f"Processing {video_id}"):
            try:
                reply = generate_counterspeech(c)
            except Exception as e:
                reply = f"[ERROR] {e}"

            all_results.append({
                "video_id": video_id,
                "comment": c,
                "counterspeech_english": reply
            })

            time.sleep(0.05)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("✅ DONE — saved to", output_file)

if __name__ == "__main__":
    main()
