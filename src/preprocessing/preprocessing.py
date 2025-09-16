import json
import re
import unicodedata
import emoji
import os
from glob import glob

# --------- Disclaimer patterns to clean ----------
DISCLAIMER_PATTERNS = [
    r"\*?disclaimer\*?:?.*",
    r"\*?clarification\*?:?.*",
    r"this video.*?not.*?political",
    r"names and terms.*?communication",
    r"this content.*?not sponsored"
]

# --------- Comment cleaning function ----------
def clean_comment(text: str) -> str:
    """
    Cleans a single YouTube comment:
    - Removes URLs, timestamps
    - Removes disclaimers
    - Strips emojis and special symbols
    - Normalizes Unicode
    - Converts to lowercase
    - Removes excess punctuation (except .  ?)
    - Collapses whitespace
    """
    text = re.sub(r"http\S+|www\S+|youtu\.be\S+", "", text)  # URLs
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)  # Timestamps

    for pattern in DISCLAIMER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = unicodedata.normalize("NFKC", text)    # Normalize unicode
    text = ''.join(c for c in text if unicodedata.category(c)[0] != "C")  # Control chars
    text = text.lower()
    text = re.sub(r"[^\w\s.,!?]", "", text)  # Remove most special chars
    text = re.sub(r"\s+", " ", text).strip()     # Collapse spaces
    return text

# --------- JSON comment cleaning ----------
def clean_json_file(input_path: str, output_path: str, hinglish_hints: set = None):
    """
    Loads a YouTube comment JSON file and writes a cleaned version.
    Each comment is cleaned and classified by language.
    Output schema:
    {
      "video_id": "...",
      "title": "...",
      "comments": [
         {"text": "...", "lang": "hinglish"},
         ...
      ]
    }
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_comments = []
    for comment in data.get("comments", []):
        cleaned = clean_comment(comment)
        lang = classify_comment(cleaned, hinglish_hints or set())
        cleaned_comments.append({
            "text": cleaned,
            "lang": lang
        })

    cleaned_data = {
        "video_id": data.get("video_id"),
        "title": clean_comment(data.get("title", "")),
        "comments": cleaned_comments
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned and classified: {os.path.basename(output_path)}")


# --------- Get video ID from filename ----------
def get_video_id_from_filename(filename: str) -> str:
    match = re.match(r"youtube_(.+?)\.json", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")

# --------- Batch clean directory ----------
def batch_clean_directory(input_dir: str, output_dir: str, hinglish_hints: set):
    os.makedirs(output_dir, exist_ok=True)
    files = glob(os.path.join(input_dir, "youtube_*.json"))

    if not files:
        print("⚠️ No matching files found in input directory.")
        return

    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            video_id = get_video_id_from_filename(filename)
            output_path = os.path.join(output_dir, f"youtube_{video_id}_clean.json")
            clean_json_file(file_path, output_path, hinglish_hints)
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")


# ----------------- LANGUAGE DETECTION HELPERS ------------------
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
INDIC_SCRIPTS_RE = re.compile(r"[\u0900-\u0D7F]")  # Devanagari to Malayalam

def detect_script(text: str) -> str:
    if DEVANAGARI_RE.search(text):
        return "hi"  # Hindi / Devanagari
    elif INDIC_SCRIPTS_RE.search(text):
        return "indic-script"
    return "latin"

def is_hinglish(text: str, hinglish_hints: set) -> bool:
    tokens = re.findall(r"\b\w+\b", text.lower())
    hint_count = sum(1 for t in tokens if t in hinglish_hints)
    return hint_count / (len(tokens) + 1e-5) > 0.15

def classify_comment(text: str, hinglish_hints: set) -> str:
    script = detect_script(text)
    if script == "latin":
        if is_hinglish(text, hinglish_hints):
            return "hinglish"
        else:
            return "english"
    elif script == "hi":
        return "hindi"
    elif script == "indic-script":
        return "indic-script"
    return "unknown"

def load_hinglish_hints(path: str) -> set:
    hints = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                hints.add(word.lower())
    return hints


# ----------------- Example usage ------------------
if __name__ == "__main__":
    input_dir = "../../data/raw"
    output_dir = "../../data/processed"
    hinglish_hints = load_hinglish_hints("../util/hinglish_words.txt")
    batch_clean_directory(input_dir, output_dir, hinglish_hints)

