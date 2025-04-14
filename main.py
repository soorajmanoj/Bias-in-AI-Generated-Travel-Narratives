import json
import os
from dotenv import load_dotenv
from src.api import get_youtube_service
from src.fetcher import get_video_details, get_video_comments

# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_ID_FILE = os.getenv("VIDEO_ID_FILE", "data/video_ids.txt")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/output/youtube_data.json")
MAX_COMMENTS = int(os.getenv("MAX_COMMENTS", 20))


def main():
    if not API_KEY:
        raise ValueError(
            "❌ API Key not found. Please set YOUTUBE_API_KEY in your .env file."
        )

    youtube = get_youtube_service(API_KEY)
    data = []

    with open(VIDEO_ID_FILE, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    for video_id in video_ids:
        print(f"Processing Video ID: {video_id}")
        title = get_video_details(youtube, video_id)
        comments = get_video_comments(youtube, video_id, MAX_COMMENTS)

        data.append({"video_id": video_id, "title": title, "comments": comments})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
