from googleapiclient.discovery import build
import json
import os
from dotenv import load_dotenv

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


def get_video_comments(youtube, video_id, max_comments=1000):
    """
    @brief Retrieves top-level comments from a YouTube video.

    @param youtube YouTube API service instance.
    @param video_id The ID of the YouTube video.
    @param max_comments Maximum number of comments to fetch (default is 1000).

    @return A list of comment strings.
    """
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_comments, 100),  # API limit per request
        textFormat="plainText",
    )

    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        request = youtube.commentThreads().list_next(request, response)

    return comments


# Load environment variables from the .env file
load_dotenv()

# Retrieve configuration values from environment variables or use default values
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Paths
VIDEO_ID_FILE = "../util/video_ids.txt"
OUTPUT_DIR = "../../data/raw"
MAX_COMMENTS = 1000


def main():
    """
    @brief Entry point of the YouTube data extraction script.

    @details
    Loads API key and video IDs from environment and file, connects to the YouTube API,
    fetches video titles and comments, and saves the results to one JSON file per video.

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

        # Structure the data
        video_data = {
            "video_id": video_id,
            "title": title,
            "comments": comments
        }

        # Save this video's data to its own JSON file
        video_file = os.path.join(OUTPUT_DIR, f"youtube_{video_id}.json")
        with open(video_file, "w", encoding="utf-8") as vf:
            json.dump(video_data, vf, indent=4, ensure_ascii=False)
        print(f"✅ Saved data for {video_id} to {video_file}")


if __name__ == "__main__":
    main()
