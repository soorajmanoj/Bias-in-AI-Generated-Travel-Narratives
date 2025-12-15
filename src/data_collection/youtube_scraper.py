from googleapiclient.discovery import build
import json
import os
from dotenv import load_dotenv
import csv

"""
@file youtube_scraper.py
@brief Helper functions and main script to fetch video metadata and comments using the YouTube API.

Provides `get_youtube_service`, `get_video_details`, `get_video_comments` and a `main()`
entry point that reads `util/video_ids.csv`, fetches titles and comments, and writes
the aggregated results to `data/raw/youtube_data.json`.
"""


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
            maxResults=min(max_comments, 100),
            textFormat="plainText",
        )

        while request and len(comments) < max_comments:
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                if len(comments) >= max_comments:
                    break

            if 'nextPageToken' in response:
                request = youtube.commentThreads().list_next(request, response)
            else:
                break
    except Exception as e:
        print(f"An error occurred while fetching comments for {video_id}: {e}")

    return comments


load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")


VIDEO_ID_FILE = "../util/video_ids.csv"
OUTPUT_DIR = "../../data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "youtube_data.json")
MAX_COMMENTS = 10000


def main():
    """
    @brief Entry point of the YouTube data extraction script.

    Loads API key and video IDs from a CSV file, connects to the YouTube API,
    fetches video titles and comments, and saves the aggregated results to a single JSON file.
    """

    if not API_KEY:
        raise ValueError("API Key not found. Please set YOUTUBE_API_KEY in your .env file.")

    youtube = get_youtube_service(API_KEY)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_videos_data = []
    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as ef:
                existing = json.load(ef)
                if isinstance(existing, list):
                    all_videos_data = existing
                    for v in existing:
                        if isinstance(v, dict) and 'video_id' in v:
                            existing_ids.add(v['video_id'])
        except Exception as e:
            print(f"Warning: could not read existing output {OUTPUT_FILE}: {e}")

    try:
        with open(VIDEO_ID_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                video_id, video_type = row
                if video_id in existing_ids:
                    print(f"Skipping Video ID: {video_id} (already present in {OUTPUT_FILE})")
                    continue

                print(f"Processing Video ID: {video_id} (Type: {video_type})")

                title = get_video_details(youtube, video_id)
                if not title:
                    print(f" Could not find title for video ID: {video_id}. Skipping.")
                    continue

                comments = get_video_comments(youtube, video_id, MAX_COMMENTS)

                video_data = {
                    "video_id": video_id,
                    "type": video_type,
                    "title": title,
                    "comments": comments
                }

                all_videos_data.append(video_data)
                existing_ids.add(video_id)
                print(f"Collected {len(comments)} comments for {video_id}.")

    except FileNotFoundError:
        print(f"Error: The file {VIDEO_ID_FILE} was not found.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as vf:
        json.dump(all_videos_data, vf, indent=4, ensure_ascii=False)

    print(f"\n🎉 Successfully saved all collected data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()