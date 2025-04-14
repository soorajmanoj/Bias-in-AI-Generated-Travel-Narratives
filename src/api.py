from googleapiclient.discovery import build


def get_youtube_service(api_key: str):
    """
    Create a YouTube API service instance.
    :param api_key: YouTube API key
    :return: YouTube API service instance
    """
    return build("youtube", "v3", developerKey=api_key)
