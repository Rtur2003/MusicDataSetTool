"""
Apple Music API Integration
Search and retrieve music information from Apple Music
"""

import requests
import jwt
import time
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta


class AppleMusicIntegration:
    """Integration with Apple Music API"""

    API_BASE_URL = "https://api.music.apple.com/v1"

    def __init__(self, key_id: Optional[str] = None,
                 team_id: Optional[str] = None,
                 private_key_path: Optional[str] = None):
        """
        Initialize Apple Music integration

        Args:
            key_id: Apple Music API Key ID
            team_id: Apple Developer Team ID
            private_key_path: Path to private key (.p8 file)
        """
        load_dotenv()

        self.key_id = key_id or os.getenv('APPLE_MUSIC_KEY_ID')
        self.team_id = team_id or os.getenv('APPLE_MUSIC_TEAM_ID')
        self.private_key_path = private_key_path or os.getenv('APPLE_MUSIC_PRIVATE_KEY_PATH')

        self.token = None
        self.token_expiry = None

        if not all([self.key_id, self.team_id, self.private_key_path]):
            print("Warning: Apple Music credentials not found. Some features may be limited.")
        else:
            try:
                self._generate_token()
            except Exception as e:
                print(f"Error generating Apple Music token: {e}")

    def _generate_token(self):
        """Generate a developer token for Apple Music API"""
        if not all([self.key_id, self.team_id, self.private_key_path]):
            return

        try:
            # Read private key
            with open(self.private_key_path, 'r') as key_file:
                private_key = key_file.read()

            # Token expires in 6 months (maximum allowed)
            expiry_time = datetime.now() + timedelta(days=180)
            self.token_expiry = expiry_time

            # Create JWT
            headers = {
                'alg': 'ES256',
                'kid': self.key_id
            }

            payload = {
                'iss': self.team_id,
                'iat': int(time.time()),
                'exp': int(expiry_time.timestamp())
            }

            self.token = jwt.encode(payload, private_key, algorithm='ES256', headers=headers)

        except FileNotFoundError:
            print(f"Private key file not found: {self.private_key_path}")
        except Exception as e:
            print(f"Error generating token: {e}")

    def _get_headers(self) -> Dict:
        """Get headers for API requests"""
        if not self.token:
            return {}

        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

    def search(self, query: str, types: List[str] = None, limit: int = 10) -> Dict:
        """
        Search Apple Music catalog

        Args:
            query: Search query
            types: List of types to search (songs, albums, artists, etc.)
            limit: Maximum results per type

        Returns:
            Dictionary with search results
        """
        if not self.token:
            return {}

        types = types or ['songs']
        types_str = ','.join(types)

        url = f"{self.API_BASE_URL}/catalog/us/search"
        params = {
            'term': query,
            'types': types_str,
            'limit': limit
        }

        try:
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error searching Apple Music: {e}")
            return {}

    def search_songs(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for songs

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of song information dictionaries
        """
        results = self.search(query, types=['songs'], limit=limit)

        if not results or 'results' not in results:
            return []

        songs = []
        for song in results['results'].get('songs', {}).get('data', []):
            attributes = song['attributes']
            songs.append({
                'id': song['id'],
                'name': attributes['name'],
                'artist': attributes['artistName'],
                'album': attributes['albumName'],
                'duration_ms': attributes.get('durationInMillis'),
                'release_date': attributes.get('releaseDate'),
                'genre': attributes.get('genreNames', []),
                'isrc': attributes.get('isrc'),
                'preview_url': attributes.get('previews', [{}])[0].get('url'),
                'artwork_url': attributes.get('artwork', {}).get('url'),
                'url': attributes.get('url')
            })

        return songs

    def get_song(self, song_id: str, storefront: str = 'us') -> Optional[Dict]:
        """
        Get detailed song information

        Args:
            song_id: Apple Music song ID
            storefront: Country code (default: us)

        Returns:
            Dictionary with song information
        """
        if not self.token:
            return None

        url = f"{self.API_BASE_URL}/catalog/{storefront}/songs/{song_id}"

        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()

            if 'data' not in data or not data['data']:
                return None

            song = data['data'][0]
            attributes = song['attributes']

            return {
                'id': song['id'],
                'name': attributes['name'],
                'artist': attributes['artistName'],
                'album': attributes['albumName'],
                'duration_ms': attributes.get('durationInMillis'),
                'release_date': attributes.get('releaseDate'),
                'genre': attributes.get('genreNames', []),
                'isrc': attributes.get('isrc'),
                'composer': attributes.get('composerName'),
                'track_number': attributes.get('trackNumber'),
                'disc_number': attributes.get('discNumber'),
                'preview_url': attributes.get('previews', [{}])[0].get('url'),
                'artwork_url': attributes.get('artwork', {}).get('url'),
                'url': attributes.get('url'),
                'explicit': attributes.get('contentRating') == 'explicit'
            }

        except requests.exceptions.RequestException as e:
            print(f"Error getting song: {e}")
            return None

    def get_album(self, album_id: str, storefront: str = 'us') -> Optional[Dict]:
        """
        Get album information

        Args:
            album_id: Apple Music album ID
            storefront: Country code

        Returns:
            Dictionary with album information
        """
        if not self.token:
            return None

        url = f"{self.API_BASE_URL}/catalog/{storefront}/albums/{album_id}"

        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()

            if 'data' not in data or not data['data']:
                return None

            album = data['data'][0]
            attributes = album['attributes']

            return {
                'id': album['id'],
                'name': attributes['name'],
                'artist': attributes['artistName'],
                'track_count': attributes['trackCount'],
                'release_date': attributes.get('releaseDate'),
                'genre': attributes.get('genreNames', []),
                'record_label': attributes.get('recordLabel'),
                'copyright': attributes.get('copyright'),
                'artwork_url': attributes.get('artwork', {}).get('url'),
                'url': attributes.get('url'),
                'is_single': attributes.get('isSingle', False)
            }

        except requests.exceptions.RequestException as e:
            print(f"Error getting album: {e}")
            return None

    def get_artist(self, artist_id: str, storefront: str = 'us') -> Optional[Dict]:
        """
        Get artist information

        Args:
            artist_id: Apple Music artist ID
            storefront: Country code

        Returns:
            Dictionary with artist information
        """
        if not self.token:
            return None

        url = f"{self.API_BASE_URL}/catalog/{storefront}/artists/{artist_id}"

        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()

            if 'data' not in data or not data['data']:
                return None

            artist = data['data'][0]
            attributes = artist['attributes']

            return {
                'id': artist['id'],
                'name': attributes['name'],
                'genre': attributes.get('genreNames', []),
                'url': attributes.get('url')
            }

        except requests.exceptions.RequestException as e:
            print(f"Error getting artist: {e}")
            return None

    def search_by_isrc(self, isrc: str) -> Optional[Dict]:
        """
        Search for a song by ISRC code

        Args:
            isrc: International Standard Recording Code

        Returns:
            Song information dictionary
        """
        results = self.search_songs(f"isrc:{isrc}", limit=1)

        if results:
            return results[0]

        return None

    def format_song_summary(self, song_info: Dict) -> str:
        """
        Format song information as a readable summary

        Args:
            song_info: Song information dictionary

        Returns:
            Formatted string summary
        """
        duration_min = song_info.get('duration_ms', 0) / 60000

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║               APPLE MUSIC TRACK INFORMATION                  ║
╠══════════════════════════════════════════════════════════════╣
║ Track: {song_info['name'][:50]:<50} ║
║ Artist: {song_info['artist'][:49]:<49} ║
║ Album: {song_info['album'][:50]:<50} ║
║ Duration: {duration_min:.2f} minutes                                      ║
║ Release Date: {song_info.get('release_date', 'N/A'):<44} ║
║ Genre: {', '.join(song_info.get('genre', []))[:50]:<50} ║
"""

        if song_info.get('isrc'):
            summary += f"║ ISRC: {song_info['isrc']:<52} ║\n"

        if song_info.get('url'):
            summary += f"║ URL: {song_info['url'][:53]:<53} ║\n"

        summary += "╚══════════════════════════════════════════════════════════════╝\n"
        return summary


# Example usage
if __name__ == "__main__":
    apple_music = AppleMusicIntegration()

    if apple_music.token:
        print("AppleMusicIntegration initialized successfully!")
        print("\nAvailable methods:")
        print("  - search_songs(query)")
        print("  - get_song(song_id)")
        print("  - get_album(album_id)")
        print("  - get_artist(artist_id)")
        print("  - search_by_isrc(isrc)")
    else:
        print("Apple Music integration not available. Please set credentials in .env file.")
        print("\nRequired credentials:")
        print("  - APPLE_MUSIC_KEY_ID")
        print("  - APPLE_MUSIC_TEAM_ID")
        print("  - APPLE_MUSIC_PRIVATE_KEY_PATH")
