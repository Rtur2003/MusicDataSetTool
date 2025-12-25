"""
Spotify API Integration
Fetch music metadata, audio features, and track information from Spotify
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

from ..utils.logger import get_logger
from ..utils.exceptions import APIError, ValidationError
from ..utils.validators import validate_positive_int

logger = get_logger(__name__)


class SpotifyIntegration:
    """Integration with Spotify Web API"""

    def __init__(self, client_id: Optional[str] = None,
                 client_secret: Optional[str] = None):
        """
        Initialize Spotify integration

        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
        """
        load_dotenv()

        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')

        if not self.client_id or not self.client_secret:
            logger.warning("Spotify credentials not found. Spotify features will be disabled.")
            self.sp = None
        else:
            try:
                auth_manager = SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                self.sp = spotipy.Spotify(auth_manager=auth_manager)
                logger.info("Spotify integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Spotify client: {str(e)}")
                self.sp = None

    def search_track(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for tracks on Spotify

        Args:
            query: Search query (track name, artist, etc.)
            limit: Maximum number of results

        Returns:
            List of track information dictionaries
            
        Raises:
            ValidationError: If parameters are invalid
            APIError: If API request fails
        """
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
            
        validate_positive_int(limit, "limit", min_value=1)
        
        if not self.sp:
            logger.warning("Spotify client not initialized")
            return []

        try:
            results = self.sp.search(q=query, type='track', limit=min(limit, 50))
            tracks = []

            for item in results['tracks']['items']:
                track_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'artists': [artist['name'] for artist in item['artists']],
                    'album': item['album']['name'],
                    'release_date': item['album']['release_date'],
                    'duration_ms': item['duration_ms'],
                    'popularity': item['popularity'],
                    'preview_url': item.get('preview_url'),
                    'external_url': item['external_urls']['spotify']
                }
                tracks.append(track_info)

            logger.debug(f"Found {len(tracks)} tracks for query: {query}")
            return tracks

        except Exception as e:
            logger.error(f"Spotify search failed for query '{query}': {str(e)}")
            raise APIError(f"Spotify search failed: {str(e)}")

    def get_track_info(self, track_id: str) -> Optional[Dict]:
        """
        Get detailed track information

        Args:
            track_id: Spotify track ID

        Returns:
            Dictionary with track information
            
        Raises:
            ValidationError: If track_id is invalid
            APIError: If API request fails
        """
        if not track_id or not track_id.strip():
            raise ValidationError("Track ID cannot be empty")
            
        if not self.sp:
            logger.warning("Spotify client not initialized")
            return None

        try:
            track = self.sp.track(track_id)

            return {
                'id': track['id'],
                'name': track['name'],
                'artists': [artist['name'] for artist in track['artists']],
                'album': track['album']['name'],
                'release_date': track['album']['release_date'],
                'duration_ms': track['duration_ms'],
                'popularity': track['popularity'],
                'explicit': track['explicit'],
                'preview_url': track.get('preview_url'),
                'external_url': track['external_urls']['spotify'],
                'isrc': track['external_ids'].get('isrc')
            }

        except Exception as e:
            logger.error(f"Failed to get track info for ID '{track_id}': {str(e)}")
            raise APIError(f"Failed to get track info: {str(e)}")

    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """
        Get Spotify's audio features for a track

        Args:
            track_id: Spotify track ID

        Returns:
            Dictionary with audio features
            
        Raises:
            ValidationError: If track_id is invalid
            APIError: If API request fails
        """
        if not track_id or not track_id.strip():
            raise ValidationError("Track ID cannot be empty")
            
        if not self.sp:
            logger.warning("Spotify client not initialized")
            return None

        try:
            features = self.sp.audio_features(track_id)[0]

            if not features:
                logger.warning(f"No audio features found for track ID: {track_id}")
                return None

            return {
                'danceability': features['danceability'],
                'energy': features['energy'],
                'key': features['key'],
                'loudness': features['loudness'],
                'mode': features['mode'],
                'speechiness': features['speechiness'],
                'acousticness': features['acousticness'],
                'instrumentalness': features['instrumentalness'],
                'liveness': features['liveness'],
                'valence': features['valence'],
                'tempo': features['tempo'],
                'duration_ms': features['duration_ms'],
                'time_signature': features['time_signature']
            }

        except Exception as e:
            logger.error(f"Failed to get audio features for ID '{track_id}': {str(e)}")
            raise APIError(f"Failed to get audio features: {str(e)}")

    def get_audio_analysis(self, track_id: str) -> Optional[Dict]:
        """
        Get detailed audio analysis from Spotify

        Args:
            track_id: Spotify track ID

        Returns:
            Dictionary with detailed audio analysis
        """
        if not self.sp:
            return None

        try:
            analysis = self.sp.audio_analysis(track_id)

            return {
                'duration': analysis['track']['duration'],
                'sample_rate': analysis['track']['sample_md5'],
                'end_of_fade_in': analysis['track']['end_of_fade_in'],
                'start_of_fade_out': analysis['track']['start_of_fade_out'],
                'loudness': analysis['track']['loudness'],
                'tempo': analysis['track']['tempo'],
                'tempo_confidence': analysis['track']['tempo_confidence'],
                'time_signature': analysis['track']['time_signature'],
                'time_signature_confidence': analysis['track']['time_signature_confidence'],
                'key': analysis['track']['key'],
                'key_confidence': analysis['track']['key_confidence'],
                'mode': analysis['track']['mode'],
                'mode_confidence': analysis['track']['mode_confidence'],
                'num_segments': len(analysis['segments']),
                'num_sections': len(analysis['sections']),
                'num_bars': len(analysis['bars']),
                'num_beats': len(analysis['beats']),
                'num_tatums': len(analysis['tatums'])
            }

        except Exception as e:
            print(f"Error getting audio analysis: {e}")
            return None

    def get_track_by_name(self, track_name: str, artist_name: Optional[str] = None) -> Optional[str]:
        """
        Get Spotify track ID by track name and optional artist

        Args:
            track_name: Name of the track
            artist_name: Optional artist name for better matching

        Returns:
            Spotify track ID or None
        """
        query = track_name
        if artist_name:
            query = f"{track_name} artist:{artist_name}"

        results = self.search_track(query, limit=1)

        if results:
            return results[0]['id']

        return None

    def get_recommendations(self, seed_tracks: List[str] = None,
                           seed_artists: List[str] = None,
                           seed_genres: List[str] = None,
                           limit: int = 10,
                           **kwargs) -> List[Dict]:
        """
        Get track recommendations based on seeds

        Args:
            seed_tracks: List of track IDs
            seed_artists: List of artist IDs
            seed_genres: List of genre names
            limit: Number of recommendations
            **kwargs: Additional parameters (target_valence, target_energy, etc.)

        Returns:
            List of recommended tracks
        """
        if not self.sp:
            return []

        try:
            results = self.sp.recommendations(
                seed_tracks=seed_tracks,
                seed_artists=seed_artists,
                seed_genres=seed_genres,
                limit=limit,
                **kwargs
            )

            tracks = []
            for item in results['tracks']:
                track_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'artists': [artist['name'] for artist in item['artists']],
                    'album': item['album']['name'],
                    'external_url': item['external_urls']['spotify']
                }
                tracks.append(track_info)

            return tracks

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def get_artist_info(self, artist_id: str) -> Optional[Dict]:
        """
        Get artist information

        Args:
            artist_id: Spotify artist ID

        Returns:
            Dictionary with artist information
        """
        if not self.sp:
            return None

        try:
            artist = self.sp.artist(artist_id)

            return {
                'id': artist['id'],
                'name': artist['name'],
                'genres': artist['genres'],
                'popularity': artist['popularity'],
                'followers': artist['followers']['total'],
                'external_url': artist['external_urls']['spotify']
            }

        except Exception as e:
            print(f"Error getting artist info: {e}")
            return None

    def get_complete_analysis(self, track_id: str) -> Dict:
        """
        Get complete analysis combining track info and audio features

        Args:
            track_id: Spotify track ID

        Returns:
            Dictionary with complete track analysis
        """
        analysis = {
            'track_info': self.get_track_info(track_id),
            'audio_features': self.get_audio_features(track_id),
            'audio_analysis': self.get_audio_analysis(track_id)
        }

        return analysis

    def format_track_summary(self, track_info: Dict, audio_features: Optional[Dict] = None) -> str:
        """
        Format track information as a readable summary

        Args:
            track_info: Track information dictionary
            audio_features: Optional audio features dictionary

        Returns:
            Formatted string summary
        """
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                  SPOTIFY TRACK INFORMATION                   ║
╠══════════════════════════════════════════════════════════════╣
║ Track: {track_info['name']:<51} ║
║ Artist(s): {', '.join(track_info['artists']):<47} ║
║ Album: {track_info['album']:<51} ║
║ Release Date: {track_info['release_date']:<44} ║
║ Duration: {track_info['duration_ms'] / 60000:.2f} minutes                                      ║
║ Popularity: {track_info['popularity']}/100                                        ║
║ URL: {track_info['external_url']:<53} ║
"""

        if audio_features:
            summary += f"""║                                                              ║
║ AUDIO FEATURES:                                              ║
║   Danceability: {audio_features['danceability']:.3f}                                      ║
║   Energy: {audio_features['energy']:.3f}                                            ║
║   Valence (Mood): {audio_features['valence']:.3f}                                  ║
║   Tempo: {audio_features['tempo']:.1f} BPM                                        ║
║   Acousticness: {audio_features['acousticness']:.3f}                                    ║
║   Instrumentalness: {audio_features['instrumentalness']:.3f}                              ║
"""

        summary += "╚══════════════════════════════════════════════════════════════╝\n"
        return summary


# Example usage
if __name__ == "__main__":
    spotify = SpotifyIntegration()

    if spotify.sp:
        print("SpotifyIntegration initialized successfully!")
        print("\nAvailable methods:")
        print("  - search_track(query)")
        print("  - get_track_info(track_id)")
        print("  - get_audio_features(track_id)")
        print("  - get_audio_analysis(track_id)")
        print("  - get_recommendations(seed_tracks, ...)")
        print("  - get_artist_info(artist_id)")
    else:
        print("Spotify integration not available. Please set credentials in .env file.")
