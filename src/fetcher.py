def get_video_details(youtube, video_id):
    """
    Fetch video details from YouTube.
    :param youtube: YouTube API service instance
    :param video_id: ID of the YouTube video
    :return: Title of the video
    """
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    items = response.get("items", [])
    if items:
        return items[0]["snippet"]["title"]
    return None


def get_video_comments(youtube, video_id, max_comments=20):
    """
    Fetch comments from a YouTube video.
    :param youtube: YouTube API service instance
    :param video_id: ID of the YouTube video
    :param max_comments: Maximum number of comments to fetch
    :return: List of comments
    """
    comments = []
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
        request = youtube.commentThreads().list_next(request, response)

    return comments
