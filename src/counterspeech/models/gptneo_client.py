import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


"""
@file gptneo_client.py
@brief Full dataset GPT-Neo-based counterspeech generator with unrestricted responses.

Uses GPT-Neo model with no safety filters to study pure, unbiased model behavior.
Generates responses without censorship or content restrictions.
"""

print("🔹 Loading EleutherAI/gpt-neo-1.3B model...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neo-1.3B",
    trust_remote_code=True,
    padding_side="left",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-1.3B",
    torch_dtype=torch.float16,
    device_map=device,
    trust_remote_code=True
)

model.eval()
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

print("✨ Model loaded successfully!")

INPUT_FILE = "../../../data/clean/filtered/merged_output.json"
PARTIAL_SAVE_FILE = "../outputs/gptneo_partial.json"
FINAL_SAVE_FILE = "../../../data/clean/filtered/gptneo_counterspeech_output_final.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

rom_hindi_comments = data.get("rom_hindi", [])
english_comments = data.get("english", [])

print(f"📌 Loaded {len(rom_hindi_comments)} Roman Hindi comments")
print(f"📌 Loaded {len(english_comments)} English comments")

SYSTEM_PROMPT = """
You are responding to comments about travel experiences. Read each comment and respond naturally based only on what's written in that specific comment. Don't bring in any external knowledge about places, cultures, or real-world context - respond purely to the words and sentiment expressed in the comment itself.

Keep your response conversational and authentic, like you're chatting with someone about their specific comment.
"""


def build_prompt(comment: str) -> str:
    """
    @brief Build a natural conversational prompt for the model.

    @param comment Original user comment string.
    @return The full prompt string sent to the model.
    """
    return f"{SYSTEM_PROMPT}\n\nSomeone said: \"{comment}\"\n\nYour response:"


def clean_output(text: str) -> str:
    """
    @brief Very minimal post-processing to preserve model biases.

    @param text Raw model output.
    @return Barely cleaned response string.
    """
    # Look for response indicators
    response_markers = ["Your response:", "Response:", "\n\n"]
    for marker in response_markers:
        if marker in text:
            text = text.split(marker, 1)[-1]
            break

    # Only basic trimming
    return text.strip()


def generate_batch(comments):
    """
    @brief Generate highly randomized counterspeech for bias analysis.

    @param comments List of comment strings.
    @return List of generated responses with high variation.
    """
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
            max_new_tokens=150,  # Very long for creative outputs
            temperature=1.5,     # Very high randomization
            top_p=0.98,          # Very broad token selection
            do_sample=True,
            repetition_penalty=1.05,  # Minimal repetition penalty
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned = [clean_output(t) for t in decoded]
    return cleaned


def batch(iterable, batch_size=12):  # Medium batch size
    """
    @brief Yield successive batches from an iterable.

    @param iterable List-like input.
    @param batch_size Batch size.
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


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
print(f"⏱️  Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

all_comments = all_comments[start_index:]

print(f"📝 Processing {len(all_comments)} comments...")

BATCH_SIZE = 12

for idx, comment_batch in enumerate(batch(all_comments, BATCH_SIZE), start=1):
    start = time.time()

    texts = [c for c, lang in comment_batch]
    replies = generate_batch(texts)

    for (orig_comment, lang), reply in zip(comment_batch, replies):
        output.append({
            "comment": orig_comment,
            "language": lang,
            "counterspeech_english": reply
        })

    with open(PARTIAL_SAVE_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    end = time.time()

    print(f"  💾 Saved batch {idx} | Total processed: {len(output)} | time: {round(end - start, 2)} seconds")

with open(FINAL_SAVE_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("🎉 COMPLETED — Final output saved!")
print(f"💾 File: {FINAL_SAVE_FILE}")

if os.path.exists(PARTIAL_SAVE_FILE):
    os.remove(PARTIAL_SAVE_FILE)
    print("🧹 Removed partial save file.")
