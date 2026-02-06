import json
import os
import time
import torch
from transformers import AutoTokenizer, pipeline

"""
@file mistral_client.py
@brief Ministral-3 counterspeech generator with batching + resume support.
"""

MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512-BF16"

print(f"🔹 Loading {MODEL_NAME}...")

# -------------------- Device --------------------

if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16     # MPS does NOT support bf16
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.bfloat16
else:
    device = "cpu"
    torch_dtype = torch.float32

# -------------------- Tokenizer --------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="left"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------- Text Generation Pipeline --------------------
# THIS is the critical fix: pipeline has the correct routing

generator = pipeline(
    task="text-generation",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=device if device != "mps" else -1,  # HF pipeline uses -1 for MPS
    torch_dtype=torch_dtype,
    trust_remote_code=True
)

print("✨ Model loaded successfully!")

# -------------------- Paths --------------------

INPUT_FILE = "../../../data/clean/filtered/merged_output.json"
PARTIAL_SAVE_FILE = "../outputs/ministral_partial.json"
FINAL_SAVE_FILE = "../../../data/clean/filtered/ministral_counterspeech_output_final.json"

# -------------------- Load Data --------------------

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

rom_hindi_comments = data.get("rom_hindi", [])
english_comments = data.get("english", [])

print(f"📌 Loaded {len(rom_hindi_comments)} Roman Hindi comments")
print(f"📌 Loaded {len(english_comments)} English comments")

# -------------------- Prompt --------------------

SYSTEM_PROMPT = (
    "You are a counterspeech generator.\n"
    "You MUST reply ONLY to the user's comment.\n\n"
    "Rules:\n"
    "- English only.\n"
    "- Stay on-topic.\n"
    "- No advice, no explanations, no unrelated content.\n"
    "- 1–2 sentences maximum.\n"
    "- Tone "
    ": sarcastic, rude, blunt, dismissive."
)

def build_prompt(comment: str) -> str:
    return (
        "<s>[INST]\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"User comment: {comment}\n"
        "[/INST]"
    )

def clean_output(text: str) -> str:
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[-1]
    return text.strip()

# -------------------- Generation --------------------

def generate_batch(comments):
    prompts = [build_prompt(c) for c in comments]

    outputs = generator(
        prompts,
        max_new_tokens=60,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        batch_size=len(prompts),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )

    # HF pipeline returns list[ list[{generated_text}] ]
    return [clean_output(o[0]["generated_text"]) for o in outputs]

def batch(iterable, batch_size=8):   # smaller batch for MPS stability
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

# -------------------- Resume Logic --------------------

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
all_comments = all_comments[start_index:]

print(f"➡️ Starting from index {start_index}")
print(f"📝 Processing {len(all_comments)} remaining comments...")
print(f"⏱️ Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# -------------------- Run --------------------

BATCH_SIZE = 8   # SAFE for Apple Silicon

for idx, comment_batch in enumerate(batch(all_comments, BATCH_SIZE), start=1):
    start = time.time()

    texts = [c for c, _ in comment_batch]
    replies = generate_batch(texts)

    for (orig_comment, lang), reply in zip(comment_batch, replies):
        output.append({
            "comment": orig_comment,
            "language": lang,
            "counterspeech_english": reply
        })

    with open(PARTIAL_SAVE_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(
        f"💾 Batch {idx} | Total processed: {len(output)} "
        f"| Time: {round(time.time() - start, 2)}s"
    )

# -------------------- Final Save --------------------

with open(FINAL_SAVE_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("🎉 COMPLETED — Final output saved!")
print(f"💾 File: {FINAL_SAVE_FILE}")

if os.path.exists(PARTIAL_SAVE_FILE):
    os.remove(PARTIAL_SAVE_FILE)
    print("🧹 Removed partial save file.")
