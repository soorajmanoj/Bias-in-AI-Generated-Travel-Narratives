import os
import csv
import time
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load API key
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# Better search queries
SEARCH_QUERIES = [
    "travel vlog India",
    "exploring India travel",
    "foreigners in India vlog",
    "Indian travel vlog",
    "backpacking India vlog"
]

def search_travel_videos(query, max_results=25):
    request = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        maxResults=max_results,
        regionCode="IN",
        relevanceLanguage="en",
        videoDuration="medium",  # avoid shorts
        order="relevance"
    )
    response = request.execute()
    return [(item['id']['videoId'], item['snippet']['channelId']) for item in response.get('items', [])]

def get_channel_country(channel_id):
    request = youtube.channels().list(
        part="snippet",
        id=channel_id
    )
    response = request.execute()
    items = response.get("items", [])
    if items:
        return items[0]["snippet"].get("country", None)
    return None

def label_country(country):
    if country is None:
        return "unknown"
    return "indian" if country.upper() == "IN" else "foreign"

def collect_and_label_videos():
    collected = []
    seen_ids = set()

    while len(collected) < 50:
        for query in SEARCH_QUERIES:
            videos = search_travel_videos(query, max_results=10)

            for video_id, channel_id in videos:
                if video_id in seen_ids:
                    continue

                country = get_channel_country(channel_id)
                label = label_country(country)

                if label != "unknown":
                    collected.append((video_id, label))
                    seen_ids.add(video_id)

                if len(collected) >= 50:
                    break
                time.sleep(0.3)
            if len(collected) >= 50:
                break

    # Save to CSV
    with open("../util/video_ids.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "type"])
        writer.writerows(collected)

    print("Saved labeled travel vlogs to /util/video_ids.csv")

if __name__ == "__main__":
    collect_and_label_videos()
