# Bias-in-AI-Generated-Travel-Narratives

## About
The goal of this project is to understand the various responses that a travel vlog receives. One of the
key aspects would be to extract the topic that’s been largely discussed in the responses, while the other
would focus on evaluating how differently large language models generate counter-speech, comparing
how Indic and Western LLMs respond to travel-related misconceptions. This project aims to provide
travel-related advice while identifying the key themes in the viewers responses.

## Project Structure

```plaintext
BIAS-IN-AI-GENERATED-TRAVEL-NARRATIVES/
├── .env.example            # Example environment file. Copy to .env and add your YouTube API key.
├── .gitignore              # Specifies files/folders for Git to ignore.
├── README.md               # This documentation file.
├── requirements.txt        # Python dependencies.
├── data
│   ├── output
│   │   └── youtube_data.json  # JSON file with fetched data.
│   └── video_ids.txt       # File listing YouTube video IDs (one per line).
├── src
│   ├── api.py              # Contains function to initialize YouTube API service.
│   └── fetcher.py          # Contains functions to fetch video details and comments.
└── main.py                 # Main script to orchestrate the data collection.
```

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
   git clone https://github.com/soorajmanoj/Bias-in-AI-Generated-Travel-Narratives.git
   cd BIAS-IN-AI-GENERATED-TRAVEL-NARRATIVES

2. **Install Dependencies:**
   python -m pip install -r requirements.txt

3. **Configure Environment Variables:**
   - Copy the provided `.env.example` file to a new file named `.env`
   - Open the `.env` file and add your own YouTube API key, for example:
     ```
     YOUTUBE_API_KEY=your_api_key_here
     ```
  
  - **Don't have a YouTube API key?**
     - Visit the official [Google Cloud Console](https://console.cloud.google.com/) and create a new project.
     - Enable the **YouTube Data API v3** for your project.
     - Go to **Credentials > Create API key** and copy it into your `.env` file.
     - Need help? Watch this beginner-friendly [YouTube tutorial on getting a YouTube API key](https://youtu.be/EPeDTRNKAVo?si=EifaTa0lCdJIXaE4).
     
4. **Details about Each File:**
   - **VIDEO_ID_FILE:** Path to the file containing YouTube video IDs (default: `data/video_ids.txt`)
   - **OUTPUT_FILE:** Path to the output JSON file (default: `data/output/youtube_data.json`)
   - **MAX_COMMENTS:** Maximum number of comments to fetch per video (default: `20`)


---

## Data Collection & Preparation

- **Obtain YouTube Video IDs:**
  - List the YouTube video IDs you want to analyze in the `data/video_ids.txt` file.
  - Ensure each video ID is on a separate line.

- **Data Fetching:**
  - The main script (`main.py`) will:
    - Read the video IDs from the file.
    - Use your YouTube API key to fetch video details and comments.
    - Save the data in `data/output/youtube_data.json`.

- **Data Verification:**
  - Verify that the data is correctly downloaded and formatted by inspecting the output JSON file.

---

## Running the Code

- **To run the project, open your terminal and run:**

  ```bash
  python main.py
- **The script will:**
   - Load environment variables.
   - Fetch the details for each video ID specified in data/video_ids.txt.
   - Save the data into the designated output file data/output/youtube_data.json.