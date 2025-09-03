import os
import re
import json
import glob
import unicodedata
from typing import Dict, Any, List, Tuple

import langid  # pip install langid


RAW_DIR = "../../data/raw"
PROCESSED_DIR = "../../data/processed"
OUT_DIR = os.path.join(PROCESSED_DIR, "clean")
UTIL_DIR = "../util"
HINGLISH_WORDLIST = os.path.join(UTIL_DIR, "hinglish_words.txt")

os.makedirs(OUT_DIR, exist_ok=True)


def load_hinglish_hints(path: str) -> set:
    """
    @brief Load Hinglish hints from a file (one word per line).
    @details Ignores blank lines and lines starting with '#'.
    @return A lowercase set of tokens.
    """
    hints = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            hints.add(line.lower())
    return hints

HINGLISH_HINTS = load_hinglish_hints(HINGLISH_WORDLIST)


def _strip_urls(text: str) -> str:
    """@brief Remove URLs."""
    return re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

def _remove_emojis(text: str) -> str:
    """@brief Remove most emoji/pictographs and joiners."""
    emoji_blocks = re.compile(
        "["                       
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002600-\U000026FF"  # misc symbols
        "]+"
    )
    text = emoji_blocks.sub("", text)
    return re.sub(r"[\u200d\uFE0E\uFE0F]", "", text)

def _normalize_ws(text: str) -> str:
    """@brief Collapse whitespace and trim ends."""
    return re.sub(r"\s+", " ", text).strip()

def _unicode_normalize(text: str) -> str:
    """@brief Unicode NFKC normalization."""
    return unicodedata.normalize("NFKC", text)

def clean_text(text: str) -> str:
    """@brief Apply standard cleaning pipeline to a comment."""
    t = _unicode_normalize(text or "")
    t = _strip_urls(t)
    t = _remove_emojis(t)
    t = _normalize_ws(t)
    return t


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

def contains_devanagari(text: str) -> bool:
    """@brief True if any Devanagari chars present."""
    return bool(DEVANAGARI_RE.search(text))

def guess_hinglish(text: str) -> bool:
    """@brief Heuristic: Latin-script text with common transliterated Hindi tokens."""
    if contains_devanagari(text):
        return False
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    hits = sum(1 for t in tokens if t in HINGLISH_HINTS)
    return hits >= 2

def detect_language(text: str) -> Tuple[str, float, str]:
    """
    @brief Detect language using langid, with Hinglish & Devanagari overrides.
    @return (lang_code, confidence, tag)
    """
    code, conf = langid.classify(text or "")
    tag = "langid"
    if contains_devanagari(text) and code == "en":
        code, tag = "hi", "devanagari-override"
    elif code == "en" and guess_hinglish(text):
        code, tag = "hinglish", "hinglish-heuristic"
    return code, float(conf), tag


def load_raw(path: str) -> Dict[str, Any]:
    """@brief Load one raw youtube_<id>.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_processed(video_id: str, payload: Dict[str, Any]) -> str:
    """@brief Save processed file as youtube_<id>_clean.json."""
    out_path = os.path.join(OUT_DIR, f"youtube_{video_id}_clean.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def process_one(raw_path: str) -> str:
    """
    @brief Process a single raw JSON into a cleaned JSON.
    @param raw_path Path to ../../data/raw/youtube_<id>.json
    @return Output path of processed JSON.
    """
    blob = load_raw(raw_path)
    video_id = blob.get("video_id") or ""
    title = blob.get("title") or ""
    comments = blob.get("comments") or []

    cleaned_items: List[Dict[str, Any]] = []
    for idx, c in enumerate(comments):
        orig = c if isinstance(c, str) else str(c)
        clean = clean_text(orig)
        lang_code, lang_conf, lang_tag = detect_language(clean)
        cleaned_items.append({
            "index": idx,
            "text_original": orig,
            "text_clean": clean,
            "lang": lang_code,
            "lang_confidence": lang_conf,
            "lang_tag": lang_tag
        })

    out_payload = {"video_id": video_id, "title": title, "comments": cleaned_items}
    out_path = save_processed(video_id, out_payload)
    print(f"✅ {video_id}: {len(cleaned_items)} comments → {out_path}")
    return out_path

def main():
    """@brief Iterate all raw files and write one processed file per raw file."""
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "youtube_*.json")))
    if not raw_files:
        raise FileNotFoundError(f"No raw files found in {RAW_DIR}")
    for rp in raw_files:
        try:
            process_one(rp)
        except Exception as e:
            print(f"⚠️  Skipping {rp}: {e}")

if __name__ == "__main__":
    main()
