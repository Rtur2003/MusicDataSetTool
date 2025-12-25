"""
YouTube API Integration
Search and retrieve video/audio information from YouTube
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import re

from ..utils.logger import get_logger
from ..utils.exceptions import APIError, ValidationError
from ..utils.validators import validate_positive_int

logger = get_logger(__name__)


class YouTubeIntegration:
    """Integration with YouTube Data API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize YouTube integration

        Args:
            api_key: YouTube Data API key
        """
        load_dotenv()

        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')

        if not self.api_key:
            logger.warning("YouTube API key not found. YouTube features will be disabled.")
            self.youtube = None
        else:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
                logger.info("YouTube integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube client: {str(e)}")
                self.youtube = None

    def search_videos(self, query: str, max_results: int = 10,
                     video_category: Optional[str] = None) -> List[Dict]:
        """
        Search for videos on YouTube

        Args:
            query: Search query
            max_results: Maximum number of results
            video_category: Optional category ID (10 for Music)

        Returns:
            List of video information dictionaries
            
        Raises:
            ValidationError: If parameters are invalid
            APIError: If API request fails
        """
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
            
        validate_positive_int(max_results, "max_results", min_value=1)
        
        if not self.youtube:
            logger.warning("YouTube client not initialized")
            return []

        try:
            search_params = {
                'q': query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': min(max_results, 50)
            }

            if video_category:
                search_params['videoCategoryId'] = video_category

            request = self.youtube.search().list(**search_params)
            response = request.execute()

            videos = []
            for item in response.get('items', []):
                video_info = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'channel_title': item['snippet']['channelTitle'],
                    'channel_id': item['snippet']['channelId'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                }
                videos.append(video_info)

            logger.debug(f"Found {len(videos)} videos for query: {query}")
            return videos

        except HttpError as e:
            logger.error(f"YouTube API HTTP error: {str(e)}")
            raise APIError(f"YouTube API error: {str(e)}")
        except Exception as e:
            logger.error(f"YouTube search failed for query '{query}': {str(e)}")
            raise APIError(f"YouTube search failed: {str(e)}")

    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """
        Get detailed video information

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with video information
            
        Raises:
            ValidationError: If video_id is invalid
            APIError: If API request fails
        """
        if not video_id or not video_id.strip():
            raise ValidationError("Video ID cannot be empty")
            
        if not self.youtube:
            logger.warning("YouTube client not initialized")
            return None

        try:
            request = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            )
            response = request.execute()

            if not response.get('items'):
                logger.warning(f"No video found with ID: {video_id}")
                return None

            video = response['items'][0]
            duration = self._parse_duration(video['contentDetails']['duration'])

            return {
                'video_id': video['id'],
                'title': video['snippet']['title'],
                'description': video['snippet']['description'],
                'channel_title': video['snippet']['channelTitle'],
                'channel_id': video['snippet']['channelId'],
                'published_at': video['snippet']['publishedAt'],
                'duration_seconds': duration,
                'duration_formatted': self._format_duration(duration),
                'view_count': int(video['statistics'].get('viewCount', 0)),
                'like_count': int(video['statistics'].get('likeCount', 0)),
                'comment_count': int(video['statistics'].get('commentCount', 0)),
                'category_id': video['snippet']['categoryId'],
                'tags': video['snippet'].get('tags', []),
                'thumbnail_url': video['snippet']['thumbnails']['high']['url'],
                'url': f"https://www.youtube.com/watch?v={video['id']}"
            }

        except HttpError as e:
            logger.error(f"YouTube API HTTP error: {str(e)}")
            raise APIError(f"YouTube API error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to get video info for ID '{video_id}': {str(e)}")
            raise APIError(f"Failed to get video info: {str(e)}")

    def search_music(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search specifically for music videos

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of music video information
        """
        # Category ID 10 is Music
        return self.search_videos(query, max_results, video_category='10')

    def extract_video_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL

        Args:
            url: YouTube URL

        Returns:
            Video ID or None
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def get_channel_info(self, channel_id: str) -> Optional[Dict]:
        """
        Get channel information

        Args:
            channel_id: YouTube channel ID

        Returns:
            Dictionary with channel information
        """
        if not self.youtube:
            return None

        try:
            request = self.youtube.channels().list(
                part='snippet,statistics',
                id=channel_id
            )
            response = request.execute()

            if not response.get('items'):
                return None

            channel = response['items'][0]

            return {
                'channel_id': channel['id'],
                'title': channel['snippet']['title'],
                'description': channel['snippet']['description'],
                'published_at': channel['snippet']['publishedAt'],
                'thumbnail_url': channel['snippet']['thumbnails']['high']['url'],
                'subscriber_count': int(channel['statistics'].get('subscriberCount', 0)),
                'video_count': int(channel['statistics'].get('videoCount', 0)),
                'view_count': int(channel['statistics'].get('viewCount', 0)),
                'url': f"https://www.youtube.com/channel/{channel['id']}"
            }

        except HttpError as e:
            print(f"YouTube API error: {e}")
            return None
        except Exception as e:
            print(f"Error getting channel info: {e}")
            return None

    def get_video_comments(self, video_id: str, max_results: int = 20) -> List[Dict]:
        """
        Get comments for a video

        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments

        Returns:
            List of comment dictionaries
        """
        if not self.youtube:
            return []

        try:
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_results,
                order='relevance'
            )
            response = request.execute()

            comments = []
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'like_count': comment['likeCount'],
                    'published_at': comment['publishedAt']
                })

            return comments

        except HttpError as e:
            print(f"YouTube API error (comments may be disabled): {e}")
            return []
        except Exception as e:
            print(f"Error getting comments: {e}")
            return []

    def _parse_duration(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration to seconds

        Args:
            duration_str: ISO 8601 duration string (e.g., 'PT4M20S')

        Returns:
            Duration in seconds
        """
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)

        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    def _format_duration(self, seconds: int) -> str:
        """
        Format duration in seconds to readable string

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"

    def format_video_summary(self, video_info: Dict) -> str:
        """
        Format video information as a readable summary

        Args:
            video_info: Video information dictionary

        Returns:
            Formatted string summary
        """
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                  YOUTUBE VIDEO INFORMATION                   ║
╠══════════════════════════════════════════════════════════════╣
║ Title: {video_info['title'][:50]:<50} ║
║ Channel: {video_info['channel_title'][:48]:<48} ║
║ Duration: {video_info['duration_formatted']:<48} ║
║ Published: {video_info['published_at'][:48]:<48} ║
║                                                              ║
║ STATISTICS:                                                  ║
║   Views: {video_info['view_count']:,}                                         ║
║   Likes: {video_info['like_count']:,}                                         ║
║   Comments: {video_info['comment_count']:,}                                      ║
║                                                              ║
║ URL: {video_info['url']:<53} ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary


# Example usage
if __name__ == "__main__":
    youtube = YouTubeIntegration()

    if youtube.youtube:
        print("YouTubeIntegration initialized successfully!")
        print("\nAvailable methods:")
        print("  - search_videos(query)")
        print("  - search_music(query)")
        print("  - get_video_info(video_id)")
        print("  - extract_video_id_from_url(url)")
        print("  - get_channel_info(channel_id)")
        print("  - get_video_comments(video_id)")
    else:
        print("YouTube integration not available. Please set API key in .env file.")
