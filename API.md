# API Documentation

## MusicAnalyzer

The main class for analyzing audio files with machine learning and API integrations.

### Initialization

```python
from src.analyzer import MusicAnalyzer

analyzer = MusicAnalyzer(model_dir='data/models/')
```

**Parameters:**
- `model_dir` (str, optional): Directory containing trained models
- `config` (AppConfig, optional): Configuration object

### Methods

#### analyze_audio_file(file_path, include_apis=False)

Perform complete analysis of an audio file.

```python
result = analyzer.analyze_audio_file('song.mp3', include_apis=True)
```

**Parameters:**
- `file_path` (str): Path to audio file
- `include_apis` (bool): Whether to include API lookups

**Returns:**
- `Dict`: Analysis results containing:
  - `file_path`: Path to analyzed file
  - `analysis_timestamp`: ISO timestamp
  - `status`: 'success' or 'error'
  - `audio_features`: Dictionary of extracted features
  - `genre`: Genre classification results
  - `mood`: Mood analysis results
  - `api_data`: API lookup results (if include_apis=True)
  - `summary`: Human-readable summary

**Raises:**
- `ValidationError`: If file path is invalid
- `AudioLoadError`: If audio file cannot be loaded
- `FeatureExtractionError`: If feature extraction fails

#### batch_analyze(file_paths, output_file=None)

Analyze multiple audio files.

```python
results = analyzer.batch_analyze(
    ['song1.mp3', 'song2.mp3'],
    output_file='results.json'
)
```

**Parameters:**
- `file_paths` (List[str]): List of audio file paths
- `output_file` (str, optional): JSON file to save results

**Returns:**
- `List[Dict]`: List of analysis results

#### compare_tracks(file_paths)

Compare multiple tracks.

```python
comparison = analyzer.compare_tracks(['track1.mp3', 'track2.mp3'])
```

**Parameters:**
- `file_paths` (List[str]): List of audio file paths (2-5 tracks)

**Returns:**
- `Dict`: Comparison data with tracks, genres, moods, BPMs, keys, durations

#### export_analysis(analysis, output_path, format='json')

Export analysis to file.

```python
analyzer.export_analysis(result, 'output.json', format='json')
```

**Parameters:**
- `analysis` (Dict): Analysis dictionary
- `output_path` (str): Output file path
- `format` (str): Export format ('json' or 'txt')

---

## AudioFeatureExtractor

Extract comprehensive audio features from music files.

### Initialization

```python
from src.features.audio_features import AudioFeatureExtractor

extractor = AudioFeatureExtractor(sample_rate=22050, n_mfcc=20)
```

**Parameters:**
- `sample_rate` (int): Target sample rate (default: 22050)
- `n_mfcc` (int): Number of MFCCs to extract (default: 20)

### Methods

#### extract_all_features(file_path)

Extract all audio features from a file.

```python
features = extractor.extract_all_features('song.mp3')
```

**Returns:**
- `Dict`: Dictionary with 68+ features including:
  - Temporal: BPM, beat count, onset strength
  - Harmonic: Key, chroma features, harmonic ratio
  - Spectral: Centroid, rolloff, bandwidth, contrast, flatness
  - MFCC: 20 coefficients with mean and std
  - Energy: RMS, loudness, dynamic range
  - Duration: Seconds, minutes, samples

---

## GenreClassifier

Deep learning model for music genre classification.

### Initialization

```python
from src.models.genre_classifier import GenreClassifier

classifier = GenreClassifier()
```

### Methods

#### train(X, y, validation_split=0.2, epochs=100, batch_size=32)

Train the genre classification model.

```python
history = classifier.train(X_features, y_labels, epochs=50)
```

**Parameters:**
- `X` (np.ndarray): Feature array (n_samples, n_features)
- `y` (np.ndarray): Label array (n_samples,)
- `validation_split` (float): Validation data split ratio
- `epochs` (int): Number of training epochs
- `batch_size` (int): Training batch size

**Returns:**
- `Dict`: Training history

#### predict(features)

Predict genre for a single audio file.

```python
result = classifier.predict(features)
# {'predicted_genre': 'rock', 'confidence': 0.87, ...}
```

**Parameters:**
- `features` (Dict): Dictionary of audio features

**Returns:**
- `Dict`: Prediction results with genre, confidence, top-3 predictions

#### save_model(model_dir, model_name='genre_classifier')

Save trained model.

```python
classifier.save_model('models/')
```

#### load_model(model_dir, model_name='genre_classifier')

Load trained model.

```python
classifier.load_model('models/')
```

---

## MoodAnalyzer

Machine learning model for music mood/emotion analysis.

### Initialization

```python
from src.models.mood_analyzer import MoodAnalyzer

analyzer = MoodAnalyzer()
```

### Methods

#### predict(features)

Predict mood for a single audio file.

```python
result = analyzer.predict(features)
# {
#     'predicted_mood': 'energetic',
#     'confidence': 0.92,
#     'valence_arousal': {'valence': 0.75, 'arousal': 0.85},
#     'emotional_quadrant': 'High Energy Positive (Happy/Excited)'
# }
```

**Parameters:**
- `features` (Dict): Dictionary of audio features

**Returns:**
- `Dict`: Mood analysis results with:
  - `predicted_mood`: Predicted mood category
  - `confidence`: Confidence score
  - `valence_arousal`: Valence and arousal scores
  - `emotional_quadrant`: Emotional quadrant description
  - `top_3_predictions`: Top 3 mood predictions

#### calculate_valence_arousal(features)

Calculate valence and arousal values.

```python
va = analyzer.calculate_valence_arousal(features)
# {'valence': 0.7, 'arousal': 0.6, 'valence_label': 'positive', ...}
```

---

## API Integrations

### SpotifyIntegration

```python
from src.integrations.spotify_integration import SpotifyIntegration

spotify = SpotifyIntegration()

# Search tracks
tracks = spotify.search_track('Bohemian Rhapsody', limit=5)

# Get track info
info = spotify.get_track_info(track_id)

# Get audio features
features = spotify.get_audio_features(track_id)
```

### YouTubeIntegration

```python
from src.integrations.youtube_integration import YouTubeIntegration

youtube = YouTubeIntegration()

# Search videos
videos = youtube.search_videos('Pink Floyd', max_results=5)

# Get video info
info = youtube.get_video_info(video_id)
```

### AppleMusicIntegration

```python
from src.integrations.apple_music_integration import AppleMusicIntegration

apple = AppleMusicIntegration()

# Search songs
songs = apple.search_songs('Stairway to Heaven', limit=5)

# Get song info
info = apple.get_song(song_id)
```

---

## Utilities

### Validators

```python
from src.utils.validators import (
    validate_file_path,
    validate_url,
    validate_positive_int,
    validate_audio_format
)

# Validate file path
path = validate_file_path('audio.mp3', must_exist=True)

# Validate URL
url = validate_url('https://example.com')

# Validate positive integer
value = validate_positive_int(10, 'limit', min_value=1)
```

### Retry Logic

```python
from src.utils.retry import retry_with_backoff, retry_on_rate_limit

@retry_with_backoff(max_retries=3, initial_delay=1.0)
def api_call():
    # Code that might fail
    pass

@retry_on_rate_limit(max_retries=3, base_delay=60.0)
def rate_limited_call():
    # API call that might be rate limited
    pass
```

### Caching

```python
from src.utils.cache import SimpleCache, cached

# Using SimpleCache
cache = SimpleCache(cache_dir='.cache', ttl_seconds=3600)
cache.set('key', {'data': 'value'})
value = cache.get('key')

# Using decorator
@cached(ttl_seconds=3600)
def expensive_operation(param):
    # Long-running computation
    return result
```

### Logging

```python
from src.utils.logger import get_logger

logger = get_logger(__name__, level=logging.INFO, log_file='app.log')
logger.info('Information message')
logger.warning('Warning message')
logger.error('Error message')
```

---

## Configuration

```python
from src.config import get_config

config = get_config()

# Access configuration
print(config.audio.sample_rate)
print(config.model.batch_size)
print(config.spotify.is_configured)
```

---

## Exception Handling

```python
from src.utils.exceptions import (
    MusicAnalyzerError,
    AudioLoadError,
    FeatureExtractionError,
    ModelError,
    APIError,
    ValidationError
)

try:
    analyzer.analyze_audio_file('song.mp3')
except AudioLoadError as e:
    print(f"Failed to load audio: {e}")
except FeatureExtractionError as e:
    print(f"Feature extraction failed: {e}")
except MusicAnalyzerError as e:
    print(f"Analysis error: {e}")
```

---

## Constants

```python
from src.utils.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_MFCC,
    GENRES,
    MOODS,
    VALENCE_AROUSAL_MAP
)

print(f"Default sample rate: {DEFAULT_SAMPLE_RATE}")
print(f"Supported genres: {GENRES}")
print(f"Supported moods: {MOODS}")
```
