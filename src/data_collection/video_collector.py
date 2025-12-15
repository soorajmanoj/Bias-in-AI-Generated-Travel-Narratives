import os
import csv
import time
from dotenv import load_dotenv
from googleapiclient.discovery import build

"""
@file video_collector.py
@brief Search YouTube for travel vlogs, collect video IDs and label channels by country.

The module provides utilities to query the YouTube Search API for travel-related
videos, determine channel country, label it as `indian`/`foreign`/`unknown`, and
append results to `util/video_ids.csv` avoiding duplicates.
"""

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

SEARCH_QUERIES = [
    "travel vlog India",
    "exploring India travel",
    "foreigners in India vlog",
    "Indian travel vlog",
    "backpacking India vlog"
]


def search_travel_videos(query, max_results=25):
    """
    @brief Search YouTube for videos matching a query.

    @param query Search query string.
    @param max_results Max results to request (<=50 enforced by API).
    @return List of tuples (videoId, channelId).
    """
    request = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        maxResults=max_results,
        regionCode="IN",
        relevanceLanguage="en",
        videoDuration="medium",
        order="relevance"
    )
    response = request.execute()
    return [(item['id']['videoId'], item['snippet']['channelId']) for item in response.get('items', [])]


def get_channel_country(channel_id):
    """
    @brief Retrieve the country code for a channel, if available.

    @param channel_id YouTube channel ID.
    @return Country code string or None.
    """
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
    """
    @brief Convert an ISO country code to a label used in the CSV.

    @param country ISO country code or None.
    @return "indian", "foreign", or "unknown".
    """
    if country is None:
        return "unknown"
    return "indian" if country.upper() == "IN" else "foreign"


def collect_and_label_videos():
    """
    @brief Main loop to collect and label videos until 50 unique items are found.

    Writes results to `util/video_ids.csv`, avoiding duplicates.
    """
    collected = []

    output_csv = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'util', 'video_ids.csv'))
    seen_ids = set()

    if os.path.exists(output_csv):
        try:
            with open(output_csv, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                for row in reader:
                    if row:
                        seen_ids.add(row[0])
        except Exception as e:
            print(f"Warning: could not read existing video_ids.csv: {e}")

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

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    mode = 'a' if os.path.exists(output_csv) else 'w'
    try:
        with open(output_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow(["video_id", "type"])
            if collected:
                writer.writerows(collected)
        print(f"Saved labeled travel vlogs to {output_csv}")
    except Exception as e:
        print(f"Error writing to {output_csv}: {e}")


if __name__ == "__main__":
    collect_and_label_videos()
