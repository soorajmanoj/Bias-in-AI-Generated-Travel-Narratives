import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import random
import re

"""
@file llama_client2.py
@brief Alternative LLaMA-based counterspeech generator with natural speech patterns.

Loads a local LLaMA model, builds prompts, generates counterspeech in batches,
supports auto-resume from partial runs, and saves final outputs.
Uses a more natural, conversational approach without strict formatting rules.
"""

print("🔹 Loading meta-llama/Llama-3.2-1B-Instruct on mps...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    trust_remote_code=True,
    padding_side="left",
)

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

INPUT_FILE = "../../../data/clean/filtered/raw/merged_output.json"
PARTIAL_SAVE_FILE = "../outputs/llama2_partial.json"
FINAL_SAVE_FILE = "../../../data/clean/filtered/llama2_counterspeech_output_final.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

rom_hindi_comments = data.get("rom_hindi", [])
english_comments = data.get("english", [])

print(f"📌 Loaded {len(rom_hindi_comments)} Roman Hindi comments")
print(f"📌 Loaded {len(english_comments)} English comments")

SYSTEM_PROMPT = """
You are responding to comments about travel experiences. Read each comment and respond naturally based only on what's written in that specific comment. Don't bring in any external knowledge about places, cultures, or real-world context - respond purely to the words and sentiment expressed in the comment itself.

CRITICAL REQUIREMENTS:
- Respond ONLY in English - no Hindi, Hinglish, or other languages
- Stay focused on the specific comment content
- Keep responses conversational but relevant
- Do not add external context or knowledge
- 1-2 sentences maximum

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
    @brief Post-process generated text to extract only the response portion.

    @param text Raw model output.
    @return Cleaned response string.
    """
    # Look for various response indicators
    response_markers = ["Your response:", "Response:", "\"", "\n\n"]
    for marker in response_markers:
        if marker in text:
            text = text.split(marker, 1)[-1]
            break

    # Clean up common artifacts
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return text.strip()


def is_english_response(text: str) -> bool:
    """
    @brief Check if response is primarily in English (not Hindi/Hinglish).

    @param text Response text to check.
    @return True if response appears to be in English.
    """
    # Count Hindi/Devanagari characters
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))

    # Count total characters
    total_chars = len(text.replace(' ', ''))

    # If more than 10% Hindi characters, likely not English
    if total_chars > 0 and (hindi_chars / total_chars) > 0.1:
        return False

    # Check for common Hindi words that shouldn't be in English responses
    hindi_words = ['hai', 'hai', 'ka', 'ke', 'ki', 'ko', 'se', 'mein', 'par', 'aur', 'bhi', 'kar', 'raha', 'rahi', 'the', 'hai']
    words = text.lower().split()
    hindi_word_count = sum(1 for word in words if word in hindi_words)

    # If more than 20% of words are common Hindi words, likely not English
    if len(words) > 0 and (hindi_word_count / len(words)) > 0.2:
        return False

    return True


def filter_response(text: str) -> str:
    """
    @brief Filter and clean the response, ensuring it's appropriate.

    @param text Raw response text.
    @return Filtered response or empty string if inappropriate.
    """
    # Check if response is in English
    if not is_english_response(text):
        return ""

    # Remove any remaining quotes or artifacts
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # Limit to reasonable length (max 200 chars for 1-2 sentences)
    if len(text) > 200:
        # Try to cut at sentence boundary
        sentences = text.split('.')
        if len(sentences) > 2:
            text = '.'.join(sentences[:2]) + '.'

    return text.strip()


def generate_batch(comments):
    """
    @brief Generate counterspeech for a list of comments using the local model.

    @param comments List of comment strings.
    @return List of generated responses.
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
            max_new_tokens=80,  # Slightly longer for more natural responses
            temperature=0.8,    # Slightly higher for more variety
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned = [clean_output(t) for t in decoded]
    filtered = [filter_response(t) for t in cleaned]
    return filtered


def batch(iterable, batch_size=16):
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

# Randomly select 10 comments from each language for testing
random.seed(42)  # For reproducible results
selected_rom_hindi = random.sample(rom_hindi_comments, min(10, len(rom_hindi_comments)))
selected_english = random.sample(english_comments, min(10, len(english_comments)))

print(f"🎯 Selected {len(selected_rom_hindi)} Roman Hindi comments")
print(f"🎯 Selected {len(selected_english)} English comments")

all_comments = (
    [(c, "rom_hindi") for c in selected_rom_hindi] +
    [(c, "english") for c in selected_english]
)

start_index = len(output)
print(f"➡️ Starting from index {start_index}")
print(f"⏱️  Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

all_comments = all_comments[start_index:]

print(f"📝 Processing remaining {len(all_comments)} comments...")

BATCH_SIZE = 16

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
