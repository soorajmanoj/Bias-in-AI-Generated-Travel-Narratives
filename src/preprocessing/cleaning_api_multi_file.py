import os
import json
import time
import sys
import argparse
import itertools
import logging
from pathlib import Path
import glob

try:
    import google.generativeai as genai
except Exception:
    genai = None

from dotenv import load_dotenv


MODEL_NAME = 'gemini-2.5-flash-lite'
DEFAULT_BATCH_SIZE = 25
DEFAULT_BATCH_DELAY_SECONDS = 15
RETRY_ATTEMPTS = 2
SKIPPED_BATCHES_FILE = "skipped_batches.json"


def load_env(root: Path):

    env_path = root / '.env'
    try:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
    except Exception as e:
        logging.warning(f"Could not load .env: {e}")


def save_skipped_batch(file_path: Path, part_file: str, video_index: int, batch_index: int, comments):
    """Append skipped batch info to skipped_batches.json for later reprocessing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = []

    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []

    entry = {
        "part_file": part_file,
        "video_index": video_index,
        "batch_index": batch_index,
        "comments": comments,
        "timestamp": int(time.time())
    }

    data.append(entry)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.warning(f"Saved skipped batch {batch_index} from video {video_index} in file {part_file}")



def discover_api_keys():
    keys = [v for k, v in os.environ.items() if k.startswith('GOOGLE_API_KEY_') and v.strip()]
    return keys


def format_time(seconds):
    if seconds < 0:
        return "0s"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
    if minutes > 0:
        return f"{int(minutes)}m {int(secs)}s"
    return f"{round(seconds)}s"


def process_comments_in_batch(model_name, comments_batch, current_api_key, attempt=1):
    """Send a batch to Gemini. Returns parsed JSON list or None on failure."""
    if genai is None:
        logging.error("google.generativeai not available. Install dependency to call API.")
        return None

    prompt = (
        "You are an expert text processor. Your task is to process a JSON array of user comments.\n\n"
        "For each comment, determine its `classification` and produce its `cleaned_text` version based on the following rules:\n\n"
        "**Classification Categories:**\n"
        "* `rom_hindi`: For comments in Romanized Hindi (Hinglish).\n"
        "* `english`: For comments primarily in English.\n"
        "* `other`: For all other languages, including Hindi in its native Devanagari script.\n\n"
        "**Cleaning Rules:**\n"
        "* **For ALL comments:** Remove emojis. Do not transliterate native scripts.\n\n"
        "Your response must be a single, valid JSON array of objects. Each object must have two keys: \"classification\" and \"cleaned_text\". "
        "Maintain the original order and include no extra text or explanations.\n\n"
        f"Comments to process:\n{json.dumps(comments_batch, ensure_ascii=False)}"
    )

    try:
        genai.configure(api_key=current_api_key)
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        result = json.loads(response.text)
        if isinstance(result, list) and len(result) == len(comments_batch):
            return result
        else:
            logging.warning("Batch returned mismatched result length or non-list response.")
            return None
    except Exception as e:
        logging.warning(f"API call failed on attempt {attempt}: {e}")
        if attempt <= RETRY_ATTEMPTS:
            time.sleep(2 ** attempt)
            return process_comments_in_batch(model_name, comments_batch, current_api_key, attempt + 1)
        return None


def merge_and_save_output(output_path: Path, all_cleaned_data: dict):

    existing = {"rom_hindi": [], "english": [], "other": []}
    if output_path.exists():
        try:
            with output_path.open('r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            logging.warning(f"Could not read existing output at {output_path}, starting fresh.")


    for k in ("rom_hindi", "english", "other"):
        existing.setdefault(k, [])
        all_cleaned_data.setdefault(k, [])


    seen = set(existing.get("rom_hindi", []) + existing.get("english", []) + existing.get("other", []))
    added = 0
    for cat, texts in all_cleaned_data.items():
        for t in texts:
            if t and t not in seen:
                existing[cat].append(t)
                seen.add(t)
                added += 1


    tmp_path = output_path.with_suffix('.tmp.json')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)
    logging.info(f"Merged {added} new comments into {output_path}")


def update_progress(progress_path: Path, last_index: int, filename: str):
    data = {
        "last_processed_index": last_index,
        "last_processed_file": filename,
        "timestamp": int(time.time())
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_progress(progress_path: Path):
    if progress_path.exists():
        try:
            return json.load(progress_path.open('r', encoding='utf-8'))
        except Exception:
            return None
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Process split YouTube data parts using Gemini and append to final output.")
    parser.add_argument('--parts-glob', default=None,
                        help='glob pattern for input part files (defaults to data/raw/youtube_data_part_*.json)')
    parser.add_argument('--start-index', type=int, default=1,
                        help='1-based index of part file to start from')
    parser.add_argument('--start-file', default=None, help='explicit filename to start from (overrides start-index)')
    parser.add_argument('--output', default=None, help='output file path (defaults to data/clean/final_API_data.json)')
    parser.add_argument('--progress-file', default=None, help='progress file storing last processed file')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--delay', type=int, default=DEFAULT_BATCH_DELAY_SECONDS)
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.resolve().parents[2]

    # File Locations
    parts_glob = args.parts_glob or str(repo_root / "Bias-in-AI-Generated-Travel-Narratives" / 'data' / 'raw' / 'youtube_data_part_*.json')
    output_path = Path(args.output) if args.output else repo_root / "Bias-in-AI-Generated-Travel-Narratives" / 'data' / 'clean' / 'final_API_data.json'
    progress_path = Path(args.progress_file) if args.progress_file else repo_root / "Bias-in-AI-Generated-Travel-Narratives" / 'data' / 'clean' / 'processing_progress.json'

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    load_env(repo_root)

    api_keys = discover_api_keys()
    if not api_keys:
        logging.error("No GOOGLE_API_KEY_<N> keys found in environment. Set them in .env or env vars.")
        sys.exit(1)

    key_cycler = itertools.cycle(enumerate(api_keys, 1))


    part_files = sorted(glob.glob(parts_glob))
    if not part_files:
        logging.error(f"No part files found with pattern: {parts_glob}")
        sys.exit(1)


    start_idx = 1
    if args.start_file:
        if args.start_file in part_files:
            start_idx = part_files.index(args.start_file) + 1
        else:

            p = str(Path(args.start_file))
            if p in part_files:
                start_idx = part_files.index(p) + 1
            else:
                logging.error(f"start-file {args.start_file} was not found among parts")
                sys.exit(1)
    else:
        start_idx = max(1, args.start_index)

    logging.info(f"Found {len(part_files)} part files. Starting from index {start_idx} ({part_files[start_idx-1] if start_idx-1 < len(part_files) else 'N/A'})")

    existing_texts = set()
    if output_path.exists():
        try:
            existing = json.load(output_path.open('r', encoding='utf-8'))
            for k in ("rom_hindi", "english", "other"):
                for t in existing.get(k, []):
                    existing_texts.add(t)
        except Exception:
            logging.warning("Could not parse existing final output; duplicates may occur.")

    files_processed = 0
    for idx, part in enumerate(part_files, start=1):
        if idx < start_idx:
            continue

        logging.info(f"Processing part {idx}/{len(part_files)}: {part}")
        files_processed += 1

        try:
            with open(part, 'r', encoding='utf-8') as f:
                videos = json.load(f)
        except Exception as e:
            logging.error(f"Could not read {part}: {e}")
            continue


        all_cleaned_data = {"rom_hindi": [], "english": [], "other": []}

        for video in videos:
            comments = video.get('comments', []) if isinstance(video, dict) else []
            total_comments = len(comments)
            if total_comments == 0:
                continue

            video_results = []
            start_time = time.time()
            for i in range(0, total_comments, args.batch_size):
                batch = comments[i:i + args.batch_size]
                key_index, current_key = next(key_cycler)
                batch_results = process_comments_in_batch(MODEL_NAME, batch, current_key)
                if batch_results:
                    video_results.extend(batch_results)
                else:
                    batch_number = i // args.batch_size + 1
                    logging.warning(f"Skipping batch {batch_number} of video due to API failure.")

                    # SAVE THE SKIPPED BATCH
                    save_skipped_batch(
                        file_path=repo_root / "Bias-in-AI-Generated-Travel-Narratives" / "data" / "clean" / SKIPPED_BATCHES_FILE,
                        part_file=str(part),
                        video_index=videos.index(video),
                        batch_index=batch_number,
                        comments=batch
                    )

                processed = min(i + args.batch_size, total_comments)
                elapsed = time.time() - start_time
                cps = processed / elapsed if elapsed > 0 else 0
                eta = ((elapsed / processed) * (total_comments - processed)) if cps > 0 else 0
                logging.info(f"Video progress: {processed}/{total_comments} | Key {key_index}/{len(api_keys)} | CPS: {cps:.2f} | ETA: {format_time(eta)}")

                if (i + args.batch_size) < total_comments:
                    time.sleep(args.delay)

            added_count = 0
            for r in video_results:
                cat = r.get('classification')
                text = (r.get('cleaned_text') or '').strip()
                if not text or cat not in all_cleaned_data:

                    if text and cat not in all_cleaned_data:
                        all_cleaned_data.setdefault('other', []).append(text)
                    continue
                if text not in existing_texts:
                    all_cleaned_data[cat].append(text)
                    existing_texts.add(text)
                    added_count += 1

            logging.info(f"Added {added_count} new unique comments from current video.")


        output_path.parent.mkdir(parents=True, exist_ok=True)
        merge_and_save_output(output_path, all_cleaned_data)


        update_progress(progress_path, idx, str(part))

    logging.info(f"Finished processing. Files processed this run: {files_processed}")


if __name__ == '__main__':
    main()
