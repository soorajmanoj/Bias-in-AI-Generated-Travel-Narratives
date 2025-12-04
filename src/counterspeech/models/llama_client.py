import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# LOAD MODEL (Llama 3.2 - 1B - MPS)
# ============================================================

print("🔹 Loading meta-llama/Llama-3.2-1B-Instruct on mps...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    trust_remote_code=True,
    padding_side="left",     # REQUIRED for decoder-only batching
)

# Llama needs a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
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
PARTIAL_SAVE_FILE = "../outputs/llama_partial.json"
FINAL_SAVE_FILE = "../outputs/llama32_counterspeech_output_final.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

rom_hindi_comments = data.get("rom_hindi", [])
english_comments = data.get("english", [])

print(f"📌 Loaded {len(rom_hindi_comments)} Roman Hindi comments")
print(f"📌 Loaded {len(english_comments)} English comments")


# ============================================================
# STRICT SYSTEM PROMPT (ANTI-HALLUCINATION)
# ============================================================

SYSTEM_PROMPT = """
You are a counterspeech generator. 
You MUST reply ONLY to the user's comment.

Rules:
- English only.
- Stay on-topic.
- No advice, no explanations, no unrelated content.
- 1–2 sentences maximum.
- Tone allowed: sarcastic, rude, blunt, dismissive.
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
# PARALLEL BATCH GENERATION (FAST + SAFE)
# ============================================================

def generate_batch(comments):
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
            max_new_tokens=60,
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
# AUTO-RESUME (LOAD PARTIAL PROGRESS)
# ============================================================

output = []

if os.path.exists(PARTIAL_SAVE_FILE):
    print("🔄 Resuming from existing partial output...")
    with open(PARTIAL_SAVE_FILE, "r") as f:
        output = json.load(f)
else:
    print("🆕 Starting fresh run...")

all_comments = (
    [(c, "rom_hindi") for c in rom_hindi_comments] +
    [(c, "english") for c in english_comments]
)

start_index = len(output)
print(f"➡️ Starting from index {start_index}")

all_comments = all_comments[start_index:]


# ============================================================
# PROCESS COMMENTS WITH SAFE BATCH-SAVE
# ============================================================

print(f"📝 Processing remaining {len(all_comments)} comments...")

BATCH_SIZE = 64  # adjust if needed (8 for safety, 24 for speed)

for idx, comment_batch in enumerate(batch(all_comments, BATCH_SIZE), start=1):

    texts = [c for c, lang in comment_batch]
    replies = generate_batch(texts)

    # Store batch output
    for (orig_comment, lang), reply in zip(comment_batch, replies):
        output.append({
            "comment": orig_comment,
            "language": lang,
            "counterspeech_english": reply
        })

    # SAVE AFTER EVERY BATCH (CRASH-PROOF)
    with open(PARTIAL_SAVE_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  💾 Saved batch {idx} | Total processed: {len(output)}")


# ============================================================
# FINAL SAVE
# ============================================================

with open(FINAL_SAVE_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("🎉 COMPLETED — Final output saved!")
print(f"💾 File: {FINAL_SAVE_FILE}")

# Remove the partial file now that we're done
if os.path.exists(PARTIAL_SAVE_FILE):
    os.remove(PARTIAL_SAVE_FILE)
    print("🧹 Removed partial save file.")
