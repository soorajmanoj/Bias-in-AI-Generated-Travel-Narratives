# Bias-in-AI-Generated-Travel-Narratives

This repository contains the full data processing, analysis, and evaluation pipeline for the project “Bias in AI-Generated Travel Narratives”.
The project studies how large language models (LLMs) generate counterspeech for toxic or biased travel-related user comments and examines toxicity bias, cultural bias, and evaluation bias using both human annotations and automated toxicity scoring tools.

---

## Project Overview

The pipeline consists of:

1. Data collection from YouTube travel-related videos and comments  
2. Data cleaning and preprocessing, including relevance filtering  
3. Counterspeech generation using multiple LLMs  
4. Toxicity evaluation using:
   - Human annotations (human scale 0–5)
   - Perspective API scores
5. Comparative analysis and visualization to identify bias patterns
---
## Repository structure:
```
Bias-in-AI-Generated-Travel-Narratives/
│
├── data/                       # All datasets (raw and processed)
│   ├── raw/                    # Raw YouTube comment data
│   └── clean/                  # Cleaned, filtered, and labeled datasets
│
├── src/                        # Source code for the full pipeline
│   ├── data_collection/        # YouTube scraping and video metadata collection
│   ├── preprocessing/          # Cleaning, filtering, and data organization
│   ├── counterspeech/          # LLM-based counterspeech generation and outputs
│   ├── analysis/               # Toxicity analysis, bias metrics, and plots
│   └── util/                   # Utility files (e.g., video ID lists)
│
├── results/                    # Topic modeling and keyword analysis outputs
├── docs/                       # Generated documentation (Doxygen)
├── requirements.txt            # Python dependencies
├── Doxyfile                    # Doxygen configuration
└── README.md                   # Project documentation
```
---
## Datasets

### Raw Data (`data/raw/`)

- **Source:** YouTube travel-related videos and comments
- **Files:**
  - `youtube_data.json`
  - `youtube_data_part_*.json`
- **Content:**
  - Video metadata
  - User comments
  - Comment text, timestamps, and identifiers

This directory contains the unfiltered corpus and includes irrelevant, neutral, toxic, and noisy content.

---

### Cleaned Data (`data/clean/`)

Key files:

- `cleaned_data_noLLM.json`  
  Text-cleaned comments after normalization and deduplication

- `final_API_data.json`  
  Fully processed dataset ready for filtering and counterspeech generation

- `processing_progress.json`  
  Tracks batch-level processing state

- `skipped_batches.json`  
  Logs skipped or failed processing batches

---

### Filtered and Labeled Data (`data/clean/filtered/`)

- `relevant.jsonl`  
  Comments selected for counterspeech generation

- `irrelevant.jsonl`  
  Comments filtered out as unsuitable

- `error.jsonl`  
  Comments that failed parsing or processing

- `llama32_counterspeech_output_final.json`  
  Counterspeech generated using LLaMA 3.2

- `qwen25_counterspeech_output_final.json`  
  Counterspeech generated using Qwen 2.5

---

### Human and Automated Toxicity Scores

Located in `src/counterspeech/outputs/`:

- `*_scored_dataset_human.json`  
  Human toxicity annotations (scale 0–5)

- `*_perspective_scores_final.json`  
  Perspective API toxicity scores for the same outputs

These paired datasets enable direct comparison between human judgment and automated evaluation.

---

## Code Modules

### Data Collection (`src/data_collection/`)

- `youtube_scraper.py`  
  Scrapes comments and metadata from YouTube

- `video_collector.py`  
  Collects and organizes video-level information

---

### Preprocessing (`src/preprocessing/`)

- `cleaning_api_multi_file.py`  
  Text cleaning and multi-file preprocessing

- `filter.py`  
  Relevance filtering of comments

- `jsonl_to_json.py`  
  Format conversion utilities

- `organize.py`  
  Sorting and dataset organization

- `scripts/split_youtube_data.py`  
  Splits large raw datasets into smaller chunks

---

### Counterspeech Generation (`src/counterspeech/`)

**Models (`models/`)**

- `llama_client.py`  
  Counterspeech generation using LLaMA 3.2

- `qwen_client.py`  
  Counterspeech generation using Qwen 2.5

**Outputs (`outputs/`)**

- Generated counterspeech datasets
- Human-labeled toxicity data
- Perspective API scoring results
- Analysis figures

---

### Analysis (`src/analysis/`)

- `analysis.py`  
  Core statistical analysis

- `final_analysis.py`  
  Aggregated results used in the paper

- `perspectiveScores.py`  
  Perspective API scoring and normalization

- `poster.py`  
  Figure generation for posters and presentations

---

### Topic and Keyword Analysis (`results/`)

- LDA topic modeling outputs
- Keyword frequency analysis
- Interactive topic visualizations

---

## Setup

Follow these steps to set up the project on your local machine:

#### 1. Clone the Repository:
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/soorajmanoj/Bias-in-AI-Generated-Travel-Narratives.git
   ```

#### 2. Create a Hugging Face API Key:
   - Go to [Hugging Face](https://huggingface.co/) and create an account if you don’t have one.
   - After logging in, visit your [Hugging Face account settings](https://huggingface.co/settings/tokens) and generate an API key.
   - Copy the generated API key.

#### 3. Install Hugging Face CLI:
   Install the Hugging Face CLI to configure and authenticate your API key:
   ```bash
   pip install huggingface-hub
   ```

#### 4. Authenticate with Hugging Face:
   Use the Hugging Face CLI to log in and store your API key:
   ```bash
   huggingface-cli login
   ```
   - When prompted, paste your Hugging Face API key.

#### 5. Install Dependencies:
   Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, manually install the required libraries using:
   ```bash
   pip install <dependency-name>
   ```

#### 6. Set Up Perspective API Key:
   - Go to the [Perspective API](https://www.perspectiveapi.com/) website and create an API key.
   - Store the API key securely:
     - Using Environment Variables:
       - Set the environment variable:
         ```bash
         export PERSPECTIVE_API_KEY="your-api-key-here"
         ```
     - Or use a `.env` file (see `.env.sample` for a template) and load it using `python-dotenv`.

---

---

## Code Reproducibility: Full End-to-End Pipeline

This section documents the **complete execution pipeline**, from raw data collection to final analysis and figure generation. Running the steps in the specified order reproduces the datasets, evaluations, and preliminary results reported in the end-term paper.


## Stage 1: Data Collection

### Step 1.1: Video Metadata Collection

**Script:** `src/data_collection/video_collector.py`

Collects and organizes travel-related YouTube video metadata that defines the scope of comment scraping.

**Output:**

```text
src/util/video_ids.csv
```

---

### Step 1.2: YouTube Comment Scraping

**Script:** `src/data_collection/youtube_scraper.py`

Uses collected video IDs to retrieve:
- User comments
- Comment metadata (IDs, timestamps)
- Video references

**Output:**

```text
data/raw/youtube_data.json
```

---

### Step 1.3: Raw Data Splitting

**Script:** `src/preprocessing/scripts/split_youtube_data.py`

Splits large raw datasets into smaller chunks to support batch processing and fault tolerance.

**Output:**

```text
data/raw/youtube_data_part_*.json
```

---

## Stage 2: Data Cleaning and Preparation

### Step 2.1: Cleaning and Normalization

**Script:** `src/preprocessing/cleaning_api_multi_file.py`

Performs:
- Text normalization
- Deduplication
- Structural validation across multiple raw files

**Primary Output:**

```text
data/clean/cleaned_data_noLLM.json
```

---

### Step 2.2: Relevance Filtering

**Script:** `src/preprocessing/filter.py`

Categorizes cleaned comments into:
- Relevant
- Irrelevant
- Error

This step determines which comments proceed to counterspeech generation.

**Output Directory:**

```text
data/clean/filtered/
```

---

### Step 2.3: Format Conversion

**Script:** `src/preprocessing/jsonl_to_json.py`

Converts filtered JSONL files into consolidated JSON format suitable for model input and evaluation.

---

### Step 2.4: Dataset Organization

**Script:** `src/preprocessing/organize.py`

Finalizes dataset structure by:
- Sorting comments
- Merging filtered outputs
- Producing analysis-ready datasets

**Final Clean Dataset:**

```text
data/clean/sorted/
```

---

## Stage 3: Counterspeech Generation

Counterspeech is generated using multiple large language models. Model-specific logic is encapsulated in separate client scripts.

### Step 3.1: LLaMA-Based Counterspeech

**Script:** `src/counterspeech/models/llama_client.py`

Generates counterspeech responses for relevant comments using LLaMA 3.2.

---

### Step 3.2: Qwen-Based Counterspeech

**Script:** `src/counterspeech/models/qwen_client.py`

Generates counterspeech responses for the same inputs using Qwen 2.5.

**Outputs (per model):**

```text
data/clean/filtered/*_counterspeech_output_final.json
```

---

## Stage 4: Toxicity Scoring and Evaluation

### Step 4.1: Automated Toxicity Scoring

**Script:** `src/analysis/perspectiveScores.py`

This step:
- Queries the Perspective API
- Normalizes toxicity scores
- Aligns automated scores with human annotations (human scale 0–5)

**Outputs:**

```text
src/counterspeech/outputs/*_perspective_scores_final.json
```

---

### Step 4.2: Human-Annotated Alignment

Human toxicity annotations are included as fixed datasets and aligned with automated scores for comparative evaluation.

**Outputs:**

```text
src/counterspeech/outputs/*_scored_dataset_human.json
```

---

## Stage 5: Analysis and Visualization

### Step 5.1: Core Statistical Analysis

**Script:** `src/analysis/analysis.py`

Computes:
- Toxicity distributions
- Human vs automated score differences
- Model-wise bias indicators

---

### Step 5.2: Final Aggregated Analysis

**Script:** `src/analysis/final_analysis.py`

Produces consolidated statistics and summaries used in the end-term paper.

---

### Step 5.3: Figure Generation

Figures for the paper and poster are generated as part of the analysis process.

**Output Directory:**

```text
src/counterspeech/outputs/figures/
```

---



## License
This project is licensed under the MIT License. See the LICENSE file for details.
