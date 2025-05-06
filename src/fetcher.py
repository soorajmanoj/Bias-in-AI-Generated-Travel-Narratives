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


def get_video_comments(youtube, video_id, max_comments=20):
    """
    @brief Retrieves top-level comments from a YouTube video.

    @param youtube YouTube API service instance.
    @param video_id The ID of the YouTube video.
    @param max_comments Maximum number of comments to fetch (default is 20).

    @return A list of comment strings.
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
