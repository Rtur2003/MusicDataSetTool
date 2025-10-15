from .analyzer import MusicAnalyzer
from .features import AudioFeatureExtractor
from .models import GenreClassifier, MoodAnalyzer
from .integrations import SpotifyIntegration, YouTubeIntegration, AppleMusicIntegration

__version__ = '1.0.0'

__all__ = [
    'MusicAnalyzer',
    'AudioFeatureExtractor',
    'GenreClassifier',
    'MoodAnalyzer',
    'SpotifyIntegration',
    'YouTubeIntegration',
    'AppleMusicIntegration'
]
