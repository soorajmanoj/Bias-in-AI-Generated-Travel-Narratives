# AGENTS.md

Guidance for AI coding agents working in the **Bias-in-AI-Generated-Travel-Narratives** project.

---

## Project Architecture

**Big Picture:** This project implements a complete ML pipeline to study bias in AI-generated counterspeech. The pipeline flows through five distinct stages:

1. **Data Collection** → YouTube comments & video metadata
2. **Preprocessing** → Cleaning, deduplication, relevance filtering
3. **Counterspeech Generation** → LLaMA 3.2 and Qwen 2.5 models (local inference)
4. **Toxicity Scoring** → Perspective API + human annotations
5. **Analysis & Visualization** → Statistical bias metrics and figures

**Data Flow Pattern:**
```
data/raw/youtube_data_part_*.json 
  → src/preprocessing/cleaning_api_multi_file.py (Gemini API)
  → data/clean/cleaned_data_noLLM.json
  → src/preprocessing/filter.py (Gemini API, relevance classification)
  → data/clean/filtered/{relevant, irrelevant, error}.jsonl
  → src/counterspeech/models/{llama,qwen}_client.py
  → data/clean/filtered/*_counterspeech_output_final.json
  → src/analysis/perspectiveScores.py (Perspective API)
  → src/counterspeech/outputs/*_perspective_scores_final.json
  → src/analysis/{analysis,final_analysis}.py
```

**Language Support:** The project handles two languages encoded in separate JSON keys:
- `"english"` — English comments
- `"rom_hindi"` — Roman Hindi (Hinglish)

These are renamed to `"hinglish"` in analysis pipelines for consistency.

---

## API Integration Points

**Multiple LLM APIs Required:**

1. **Google Generative AI (Gemini)** — Text cleaning & relevance filtering
   - Model: `gemini-2.5-flash-lite`
   - Environment: Multiple keys via `GOOGLE_API_KEY_*` pattern (e.g., `GOOGLE_API_KEY_8`)
   - Used in: `cleaning_api_multi_file.py`, `filter.py`
   - Batch processing: Default batch size = 25, delay = 15 seconds between batches
   - Fault tolerance: Tracks failed batches in `data/clean/skipped_batches.json`

2. **Perspective API** — Toxicity scoring
   - Environment: `PERSPECTIVE_API_KEY`
   - Rate limiting: Auto-backoff on HTTP 429 with 3-second sleep
   - Attributes: `TOXICITY`, `SEVERE_TOXICITY`, `INSULT`, `PROFANITY`, `IDENTITY_ATTACK`
   - Checkpointing: Progress saved every 25 items in `perspectiveScores.py`

3. **Hugging Face** — Local model inference
   - Models: `meta-llama/Llama-3.2-1B-Instruct`, `Qwen` variants
   - Device: Auto-detects MPS (Apple Silicon) or falls back to CPU
   - Requires: `huggingface-cli login` before inference

---

## Critical Workflows & Entry Points

**Stage-by-stage execution commands (from project root):**

```bash
# 1. Data collection
python src/data_collection/video_collector.py  # → util/video_ids.csv
python src/data_collection/youtube_scraper.py   # → data/raw/youtube_data.json

# 2. Preprocessing
python src/preprocessing/scripts/split_youtube_data.py  # → youtube_data_part_*.json
python src/preprocessing/cleaning_api_multi_file.py      # → cleaned_data_noLLM.json
python src/preprocessing/filter.py                       # → filtered/{relevant,irrelevant,error}.jsonl
python src/preprocessing/jsonl_to_json.py
python src/preprocessing/organize.py                      # → sorted/

# 3. Counterspeech generation (local, GPU-intensive)
python src/counterspeech/models/llama_client.py           # → llama32_counterspeech_output_final.json
python src/counterspeech/models/qwen_client.py            # → qwen25_counterspeech_output_final.json

# 4. Toxicity scoring
python src/analysis/perspectiveScores.py                  # → *_perspective_scores_final.json

# 5. Analysis
python src/analysis/analysis.py                           # Exploratory figures
python src/analysis/final_analysis.py                     # Paper results
```

**Resume from failures:**
- Cleaning API batches: Auto-resumes from `processing_progress.json`
- Failed batches: Recorded in `skipped_batches.json`, can be reprocessed

---

## Key Data Structures & Conventions

**JSON Schemas:**

1. **Raw YouTube Data** (`youtube_data_part_*.json`):
   ```json
   {
     "video_id": "...",
     "title": "...",
     "comments": ["comment_text", ...]
   }
   ```

2. **Cleaned Data** (`cleaned_data_noLLM.json`):
   ```json
   {
     "rom_hindi": [{"comment": "...", "language": "rom_hindi"}, ...],
     "english": [{"comment": "...", "language": "english"}, ...]
   }
   ```

3. **Counterspeech Output**:
   ```json
   [
     {
       "comment": "original comment",
       "lang": "rom_hindi",
       "counterspeech": "generated response",
       "model": "llama32" or "qwen25"
     }
   ]
   ```

4. **Perspective Scored Data**:
   ```json
   [
     {
       "comment": "...",
       "lang": "...",
       "counterspeech": "...",
       "perspective_scores": {
         "TOXICITY": 0.xx,
         "SEVERE_TOXICITY": 0.xx,
         ...
       }
     }
   ]
   ```

**File Naming Convention:**
- Output files include model name: `llama32_*`, `qwen25_*`
- Filtered outputs use `.jsonl` (JSON Lines) format for streaming
- Progress tracking: `processing_progress.json`, `skipped_batches.json`

---

## Project-Specific Patterns

**Environment Variable Discovery:**
- `cleaning_api_multi_file.py` discovers all `GOOGLE_API_KEY_*` env vars for parallel API calls
- Multiple API keys enable fault tolerance (retry with next key on failure)

**Doxygen Documentation:**
- All modules use `@brief` and `@param` docstring conventions (see `Doxyfile`)
- Generated HTML docs in `docs/html/`

**Batch Processing Idiom:**
- Files split into `youtube_data_part_*.json` to support checkpointed batch processing
- Each part is processed independently; progress tracked per-batch
- Failed batches recorded separately for reprocessing

**GPU Device Selection:**
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```
This is **macOS-specific** (Apple Silicon detection). Windows/Linux agents should modify.

---

## Common Gotchas & Important Details

1. **Relative Imports:** Most scripts use relative paths like `../../data/clean/...`
   - **Always run from the script's directory**, not project root
   - Or modify `INPUT_FILE`, `OUTPUT_FILE` constants if running from elsewhere

2. **Environment Files:**
   - Cleaning uses `.env` files loaded via `dotenv`
   - Missing API keys cause silent failures in some modules; always check logs

3. **Perspective API Rate Limits:**
   - Default: 1 request per 1-2 seconds
   - `perspectiveScores.py` auto-sleeps on 429; don't remove the sleep logic

4. **LLaMA Model Loading:**
   - **First run downloads ~6GB model** from Hugging Face; requires `huggingface-cli login`
   - Uses `torch_dtype=torch.float16` to reduce memory; don't change to float32 without reason
   - Generates in batch mode with `max_length` set per model config

5. **Deduplication:**
   - `cleaning_api_multi_file.py` performs text-level deduplication
   - Prevent re-running on already-cleaned data to avoid data loss

6. **Analysis expects two languages:**
   - `analysis.py` filters to `hinglish` and `english` only
   - Other languages in data will be silently dropped
   - `rom_hindi` is automatically renamed to `hinglish` during analysis

---

## Testing & Validation

- **No unit tests present** in codebase; validation is manual via data inspection
- **Recommended checks when modifying pipelines:**
  - Verify output JSON is valid before downstream consumption
  - Spot-check `cleaned_data_noLLM.json` for deduplication errors
  - Validate toxicity scores fall in [0, 1] range from Perspective API
  - Ensure counterspeech model generation uses correct prompt format

---

## Documentation & References

- **README.md** — Full pipeline walkthrough with setup instructions
- **Doxygen comments** — Inline documentation in Python modules (generated to `docs/html/`)
- **Data structure details** — See README sections "Datasets" and "Code Modules"

