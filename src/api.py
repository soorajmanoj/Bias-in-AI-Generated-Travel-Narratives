from googleapiclient.discovery import build


def get_youtube_service(api_key: str):
    """
    @brief Creates a YouTube API service instance.

    @param api_key YouTube Data API v3 key.

    @return An authorized YouTube API service object.
    """
    return build("youtube", "v3", developerKey=api_key)
