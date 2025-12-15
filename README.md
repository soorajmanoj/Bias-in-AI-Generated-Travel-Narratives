# Bias-in-AI-Generated-Travel-Narratives

This repository contains code, preprocessing pipelines, counterspeech outputs,
and analysis used to investigate model-produced counterspeech for travel-related
YouTube comments. The README below explains how to reproduce the preliminary
results and poster figures for the endterm paper.

---

## Quick overview

- Main code areas:
  - Data collection: src/data_collection/youtube_scraper.py, src/data_collection/video_collector.py
  - Preprocessing: src/preprocessing/ (filtering, cleaning, jsonljson tools)
  - Counterspeech model clients & outputs: src/counterspeech/models/ and src/counterspeech/outputs/
  - Analysis & figures: src/analysis/poster.py

---

## Environment & dependencies

- Python 3.8+ recommended.
- Install dependencies with:

`ash
python -m pip install -r requirements.txt
`

It is recommended to use a virtual environment:

`ash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
`

Key packages used (listed in equirements.txt): pandas, matplotlib, scikit-learn,
bertopic, google-api-python-client, python-dotenv, nltk, pyldavis, and others.

---

## API keys and environment variables

- Create a .env file in the project root (or copy .env.example) and add:

`
YOUTUBE_API_KEY=your_api_key_here
# Optional: keys for Perspective API or other services used by preprocessing
# PERSPECTIVE_API_KEY=your_key_here
`

- To obtain a YouTube API key: enable the YouTube Data API v3 in Google Cloud
  and create an API key under the project's Credentials page.

Note: some counterspeech model scripts may call external LLM services or
require local model checkpoints; these are optional for reproducing figures
because precomputed model outputs are included in the repository.

---

## Included datasets and files

- data/raw/ - raw YouTube JSON files (multiple parts are present in the repo).
- data/clean/ - cleaned and merged datasets used in analysis (e.g. inal_API_data.json).
- src/counterspeech/outputs/ - precomputed counterspeech and Perspective outputs used
  by poster.py:
  - llama32_scored_dataset_human.json
  - qwen25_scored_dataset_human.json
  - llama32_perspective_scores_final.json
  - qwen25_perspective_scores_final.json

If you prefer to re-run the YouTube data yourself, see the section below. If you want to
skip heavy model inference, use the provided outputs in src/counterspeech/outputs/.

---

## Reproducing the pipeline (recommended order)

1) (Optional) Fetch raw YouTube data

`ash
# from repository root
python src/data_collection/youtube_scraper.py
# or
python src/data_collection/video_collector.py
`

These scripts read video IDs from data/video_ids.txt (or data/video_ids.csv),
use YOUTUBE_API_KEY from .env, and write raw JSON to data/raw/ or the
configured output path.

2) Preprocess and clean

Use the scripts in src/preprocessing/ to filter, clean, and convert jsonl to
merged JSON. Example commands:

`ash
python src/preprocessing/scripts/split_youtube_data.py
python src/preprocessing/filter.py
python src/preprocessing/jsonl_to_json.py
python src/preprocessing/organize.py
`

Note: some preprocessing scripts may require API keys if they call external
scoring or cleaning APIs. The cleaned files in data/clean/ can be used to
skip this step.

3) (Optional / advanced) Generate counterspeech and human scores

- Running the local model clients (LLaMA / Qwen) may require large model
  downloads, GPU resources, and additional runtime packages. See
  src/counterspeech/models/llama_client.py and qwen_client.py.
- To avoid heavy compute, use the precomputed files in
  src/counterspeech/outputs/ that are included with this repository.

4) Generate poster / analysis figures

The script src/analysis/poster.py reads files from src/counterspeech/outputs/
and writes PNG figures to src/analysis/figures/ (created automatically).

Run the script from the analysis directory so the relative paths match:

`ash
cd src/analysis
python poster.py
`

This will produce files such as ig1_distribution_human.png,
ig_perspective_diff_hist.png, and others under src/analysis/figures/.

---

## Suggested minimal reproducible flow (fast)

If you only want to reproduce the key figures in the deliverable, do this:

1. Ensure dependencies are installed.
2. Ensure src/counterspeech/outputs/ contains the four JSON files listed above
   (these are included in the repo).
3. Run poster.py as shown in the previous section.

This avoids re-downloading YouTube data and re-running model inference.

---

## Notes, caveats and troubleshooting

- Scripts use relative paths; run them from the directory assumed by the script
  (the README commands show the recommended working directory).
- If you plan to re-run model-based counterspeech generation, ensure you have
  the compute resources and storage for model weights.
- If an external API fails (rate limits or missing key), use the included
  outputs to continue analysis.

---

## Contact

If you have issues reproducing the results or want help adapting the pipeline,
open an issue in the repository or contact the project owner.

---

*Prepared for the endterm paper reproducibility deliverable.*
