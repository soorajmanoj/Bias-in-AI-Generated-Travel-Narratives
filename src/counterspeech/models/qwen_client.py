import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# LOAD MODEL
# ============================================================

print("🔹 Loading Qwen2.5-3B-Instruct on mps...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True,
    padding_side="left",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map=device,
    trust_remote_code=True
)

model.eval()
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

print("✨ Model loaded successfully!")


# ============================================================
# LOAD INPUT FILE
# ============================================================

INPUT_FILE = "../../../data/clean/filtered/merged_output.json"
PARTIAL_SAVE_FILE = "../outputs/qwen25_partial.json"
FINAL_SAVE_FILE = "../outputs/qwen25_counterspeech_output.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

rom_hindi_comments = data.get("rom_hindi", [])
english_comments = data.get("english", [])

print(f"📌 Loaded {len(rom_hindi_comments)} Roman Hindi comments")
print(f"📌 Loaded {len(english_comments)} English comments")


# ============================================================
# SYSTEM PROMPT (STRICT ANTI-HALLUCINATION)
# ============================================================

SYSTEM_PROMPT = """
You MUST reply ONLY to the user's comment.
Rules:
- English only.
- Stay on-topic.
- No advice, explanations, or unrelated content.
- 1–2 sentences.
- Tone: sarcastic, rude, blunt, dismissive.
"""
def build_prompt(comment: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"[COMMENT]: {comment}\n"
        f"[RESPONSE]:"
    )


# ============================================================
# CLEAN OUTPUT
# ============================================================

def clean_output(text: str) -> str:
    if "[RESPONSE]:" in text:
        text = text.split("[RESPONSE]:", 1)[-1]

    if "User:" in text:
        text = text.split("User:", 1)[0]

    if "[COMMENT]:" in text:
        text = text.split("[COMMENT]:", 1)[0]

    return text.strip()


# ============================================================
# BATCH GENERATION
# ============================================================

def generate_counterspeech_batch(comments):
    prompts = [build_prompt(c) for c in comments]

    batch_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=40,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned = [clean_output(t) for t in decoded]
    return cleaned


def batch(iterable, batch_size=16):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


# ============================================================
# AUTO-RESUME SUPPORT
# ============================================================

output = []

if os.path.exists(PARTIAL_SAVE_FILE):
    print("🔄 Partial save detected — resuming from last progress...")
    with open(PARTIAL_SAVE_FILE, "r") as f:
        output = json.load(f)
else:
    print("🆕 Starting fresh run...")

# Flatten comments
all_comments = (
    [(c, "rom_hindi") for c in rom_hindi_comments] +
    [(c, "english") for c in english_comments]
)

# Skip already processed comments
start_index = len(output)
print(f"⏩ Resuming from comment #{start_index}")

all_comments = all_comments[start_index:]


# ============================================================
# PROCESS & BATCH-SAVE
# ============================================================

print(f"📝 Processing {len(all_comments)} remaining comments...")

BATCH_SIZE = 16

for idx, comment_batch in enumerate(batch(all_comments, BATCH_SIZE), start=1):

    start = time.time()

    texts = [c for c, lang in comment_batch]
    replies = generate_counterspeech_batch(texts)

    # Append new batch to full output
    for (orig_comment, lang), reply in zip(comment_batch, replies):
        output.append({
            "comment": orig_comment,
            "language": lang,
            "counterspeech_english": reply
        })

    # SAVE AFTER EVERY BATCH 💾
    with open(PARTIAL_SAVE_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    end = time.time()
    print(f"  💾 Saved batch {idx} | Total processed: {len(output)} | time: , {round(end - start, 2)}, seconds")


# ============================================================
# FINAL SAVE
# ============================================================

with open(FINAL_SAVE_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("🎉 ALL DONE — Final file saved!")
print(f"💾 Final saved at: {FINAL_SAVE_FILE}")

# Remove partial file since job completed successfully
if os.path.exists(PARTIAL_SAVE_FILE):
    os.remove(PARTIAL_SAVE_FILE)
    print("🧹 Cleaned partial save file.")
