# Bias-in-AI-Generated-Travel-Narratives

## About
The goal of this project is to understand the various responses that a travel vlog receives. One of the
key aspects would be to extract the topic that’s been largely discussed in the responses, while the other
would focus on evaluating how differently large language models generate counter-speech, comparing
how Indic and Western LLMs respond to travel-related misconceptions. This project aims to provide
travel-related advice while identifying the key themes in the viewers responses.

## Project Structure

|--- BIAS-IN-AI-GENERATED-TRAVEL-NARRATIVES
|    |--- .env.example
|    |--- .gitignore
|    |--- README.md
|    |--- requirements.txt
|    |--- data
|    |    |--- output
|    |    |    |--- youtube_data.json
|    |    |    |--- video_ids.txt
|    |
|    |--- src
|         |--- api.py
|         |--- fetcher.py
|         |--- main.py

---

## Requirements

- **Programming Language:** Python 3.8+  
- **Dependencies:**  
  - `google-api-python-client==2.121.0`  
  - `python-dotenv==1.0.1`  
- These dependencies are listed in `requirements.txt` and can be installed using pip.

---

## Setup & Installation

1. **Clone the Repository:**  
   Open your terminal and run:  
   ```bash
   git clone https://github.com/yourusername/BIAS-IN-AI-GENERATED-TRAVEL-NARRATIVES.git
   cd BIAS-IN-AI-GENERATED-TRAVEL-NARRATIVES

2. **Install Dependencies:**
   python -m pip install -r requirements.txt

3. **Configure Environment Variables:**
    Copy the provided .env.example file to a new file named .env:
    Open the .env file and add your own YouTube API key:
    "YOUTUBE_API_KEY=your_api_key_here"

4. **Deatils about each file:**
   VIDEO_ID_FILE: Path to the file containing YouTube video IDs (default: data/video_ids.txt)
   OUTPUT_FILE: Path to the output JSON file (default: data/output/youtube_data.json)
   MAX_COMMENTS: Maximum number of comments to fetch per video (default: 20)

## Data Collection & Preparation

Obtain YouTube Video IDs:

    List the YouTube video IDs you want to analyze in the data/video_ids.txt file.

    Ensure each video ID is on a separate line.

Data Fetching:

    The main script (main.py) will:

       Read the video IDs from the file.

       Use your YouTube API key to fetch video details and comments.

       Save the data in data/output/youtube_data.json.

Data Verification:

     Verify that the data is correctly downloaded and formatted by inspecting the output JSON file.

## Running the code:
   Command to run the project:
   python main.py
   The script will load environment variables, fetch the details for each video ID specified in data/video_ids.txt, and save the data into the designated output file data/output/youtube_data.json.