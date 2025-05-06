import json
import os
from dotenv import load_dotenv
from src.api import get_youtube_service
from src.fetcher import get_video_details, get_video_comments

# Load environment variables from the .env file
load_dotenv()

# Retrieve configuration values from environment variables or use default values
API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_ID_FILE = os.getenv("VIDEO_ID_FILE", "data/video_ids.txt")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/output/youtube_data.json")
MAX_COMMENTS = int(os.getenv("MAX_COMMENTS", 20))  # Default: 20 comments per video

def main():
    """
    @brief Entry point of the YouTube data extraction script.

    @details
    Loads API key and video IDs from environment and file, connects to the YouTube API, 
    fetches video titles and comments, and saves the results to a JSON file.

    @exception ValueError if API key is missing.

    @return None
    """
    # Ensure that the API key is set
    if not API_KEY:
        raise ValueError("❌ API Key not found. Please set YOUTUBE_API_KEY in your .env file.")

    # Initialize YouTube API client
    youtube = get_youtube_service(API_KEY)

    # Initialize a list to hold data for each video
    data = []

    # Read video IDs from the input file
    with open(VIDEO_ID_FILE, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]  # Remove whitespace and ignore empty lines

    # Process each video ID
    for video_id in video_ids:
        print(f"Processing Video ID: {video_id}")
        
        # Fetch the video title
        title = get_video_details(youtube, video_id)
        
        # Fetch the top comments (up to MAX_COMMENTS)
        comments = get_video_comments(youtube, video_id, MAX_COMMENTS)

        # Append the structured data to the list
        data.append({
            "video_id": video_id,
            "title": title,
            "comments": comments
        })

    # Write the collected data to the output JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Data saved to {OUTPUT_FILE}")

# Entry point of the script
if __name__ == "__main__":
    main()
