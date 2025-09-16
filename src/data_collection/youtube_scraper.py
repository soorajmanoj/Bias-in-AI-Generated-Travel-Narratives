from googleapiclient.discovery import build
import json
import os
from dotenv import load_dotenv
import csv  # Added for CSV file reading


def get_youtube_service(api_key: str):
    """
    @brief Creates a YouTube API service instance.

    @param api_key YouTube Data API v3 key.

    @return An authorized YouTube API service object.
    """
    return build("youtube", "v3", developerKey=api_key)


def get_video_details(youtube, video_id):
    """
    @brief Fetches video title from the YouTube API.

    @param youtube YouTube API service instance.
    @param video_id The ID of the YouTube video.

    @return The title of the video if found, otherwise None.
    """
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    items = response.get("items", [])
    if items:
        return items[0]["snippet"]["title"]
    return None


def get_video_comments(youtube, video_id, max_comments=10000):
    """
    @brief Retrieves top-level comments from a YouTube video.

    @param youtube YouTube API service instance.
    @param video_id The ID of the YouTube video.
    @param max_comments Maximum number of comments to fetch.

    @return A list of comment strings.
    """
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),  # API limit per request is 100
            textFormat="plainText",
        )

        while request and len(comments) < max_comments:
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                if len(comments) >= max_comments:
                    break
            # Check for a nextPageToken to continue pagination
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list_next(request, response)
            else:
                break  # No more pages
    except Exception as e:
        print(f"An error occurred while fetching comments for {video_id}: {e}")

    return comments


# Load environment variables from the .env file
load_dotenv()

# Retrieve configuration values from environment variables or use default values
API_KEY = os.getenv("YOUTUBE_API_KEY")

# --- CONFIGURATION UPDATED ---
# Paths
VIDEO_ID_FILE = "../util/video_ids.csv"  # Changed to read from .csv
OUTPUT_DIR = "../../data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "youtube_data.json")  # Single output file
MAX_COMMENTS = 10000  # Updated to 10,000


def main():
    """
    @brief Entry point of the YouTube data extraction script.

    @details
    Loads API key and video IDs from a CSV file, connects to the YouTube API,
    fetches video titles and comments, and saves the aggregated results to a single JSON file.

    @exception ValueError if API key is missing.

    @return None
    """
    # Ensure that the API key is set
    if not API_KEY:
        raise ValueError("❌ API Key not found. Please set YOUTUBE_API_KEY in your .env file.")

    # Initialize YouTube API client
    youtube = get_youtube_service(API_KEY)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # This list will store all the data before writing to a single file
    all_videos_data = []

    # Read video IDs and types from the input CSV file
    try:
        with open(VIDEO_ID_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row ('video_id,type')

            # Process each row in the CSV
            for row in reader:
                video_id, video_type = row
                print(f"Processing Video ID: {video_id} (Type: {video_type})")

                # Fetch the video title
                title = get_video_details(youtube, video_id)
                if not title:
                    print(f"⚠️  Could not find title for video ID: {video_id}. Skipping.")
                    continue

                # Fetch the top comments (up to MAX_COMMENTS)
                comments = get_video_comments(youtube, video_id, MAX_COMMENTS)

                # Structure the data as requested
                video_data = {
                    "video_id": video_id,
                    "type": video_type,  # Added 'type' from the CSV
                    "title": title,
                    "comments": comments
                }

                # Add this video's data to our master list
                all_videos_data.append(video_data)
                print(f"✅ Collected {len(comments)} comments for {video_id}.")

    except FileNotFoundError:
        print(f"❌ Error: The file {VIDEO_ID_FILE} was not found.")
        return

    # Save the aggregated data to a single JSON file after the loop
    with open(OUTPUT_FILE, "w", encoding="utf-8") as vf:
        json.dump(all_videos_data, vf, indent=4, ensure_ascii=False)

    print(f"\n🎉 Successfully saved all collected data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()