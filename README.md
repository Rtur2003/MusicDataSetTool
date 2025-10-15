# 🎵 Audio Analyzer

**A comprehensive music analysis tool combining machine learning, audio signal processing, and streaming platform integrations.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Features

### 🎼 Audio Feature Extraction
- **Tempo & Rhythm**: BPM, beat tracking, onset detection, tempogram
- **Harmony & Key**: Key estimation, chroma features (STFT, CQT, CENS), harmonic/percussive separation
- **Spectral Analysis**: Spectral centroid, rolloff, bandwidth, contrast, flatness
- **MFCCs**: 20 Mel-frequency cepstral coefficients with statistics
- **Energy & Dynamics**: RMS energy, loudness (dB), dynamic range
- **Time Features**: Duration, sample rate, total samples

### 🎸 Genre Classification
- Deep learning model (TensorFlow/Keras) with 5 layers
- Supports 10 genres: Rock, Pop, Jazz, Classical, Hip-Hop, Electronic, Country, Blues, Reggae, Metal
- Confidence scoring and top-3 predictions
- Rule-based fallback when model not trained

### 😊 Mood Analysis
- Machine learning model (Random Forest) with valence-arousal framework
- 8 mood categories: Happy, Sad, Energetic, Calm, Angry, Romantic, Melancholic, Uplifting
- Emotional quadrant mapping (Russell's circumplex model)
- Valence (positive/negative) and arousal (energy) scoring

### 🎵 Streaming Platform Integrations
- **Spotify**: Track search, audio features, audio analysis, recommendations
- **YouTube**: Video search, metadata retrieval, channel info, comments
- **Apple Music**: Song search, track info, album/artist data, ISRC lookup

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Rtur2003/audio-analyzer.git
cd audio-analyzer
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API credentials**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# - SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
# - YOUTUBE_API_KEY
# - APPLE_MUSIC_KEY_ID, APPLE_MUSIC_TEAM_ID, APPLE_MUSIC_PRIVATE_KEY_PATH
```

---

## 🎯 Usage

### Basic Usage

```python
from src.analyzer import MusicAnalyzer

# Initialize analyzer
analyzer = MusicAnalyzer()

# Analyze a single audio file
result = analyzer.analyze_audio_file('path/to/music.mp3', include_apis=True)

# Print summary
print(result['summary'])

# Export results
analyzer.export_analysis(result, 'output.json', format='json')
```

### Command Line Interface

```bash
# Analyze a single file
python src/analyzer.py path/to/music.mp3

# Output will be saved as music_analysis.json
```

### Batch Analysis

```python
from src.analyzer import MusicAnalyzer

analyzer = MusicAnalyzer()

# Analyze multiple files
file_paths = [
    'music/song1.mp3',
    'music/song2.mp3',
    'music/song3.mp3'
]

results = analyzer.batch_analyze(file_paths, output_file='batch_results.json')
```

### Compare Tracks

```python
# Compare 2-5 tracks
comparison = analyzer.compare_tracks([
    'track1.mp3',
    'track2.mp3',
    'track3.mp3'
])

print(comparison)
```

### Streaming Platform Analysis

```python
# Analyze from Spotify URL
spotify_result = analyzer.analyze_from_url(
    'https://open.spotify.com/track/xxxxx',
    platform='spotify'
)

# Analyze from YouTube URL
youtube_result = analyzer.analyze_from_url(
    'https://www.youtube.com/watch?v=xxxxx',
    platform='youtube'
)
```

---

## 🧪 Individual Components

### Audio Feature Extraction

```python
from src.features import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
features = extractor.extract_all_features('music.mp3')

print(f"BPM: {features['bpm']}")
print(f"Key: {features['estimated_key']}")
print(f"Duration: {features['duration_minutes']} min")
```

### Genre Classification

```python
from src.models import GenreClassifier

classifier = GenreClassifier()

# Train model (if you have labeled data)
# X = feature_array, y = labels
# classifier.train(X, y, epochs=100)

# Predict
prediction = classifier.predict(features)
print(f"Genre: {prediction['predicted_genre']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Mood Analysis

```python
from src.models import MoodAnalyzer

mood_analyzer = MoodAnalyzer()

# Predict mood
mood_result = mood_analyzer.predict(features)

print(f"Mood: {mood_result['predicted_mood']}")
print(f"Valence: {mood_result['valence_arousal']['valence']:.2f}")
print(f"Arousal: {mood_result['valence_arousal']['arousal']:.2f}")
```

### Spotify Integration

```python
from src.integrations import SpotifyIntegration

spotify = SpotifyIntegration()

# Search tracks
results = spotify.search_track('Bohemian Rhapsody', limit=5)

# Get audio features
track_id = results[0]['id']
audio_features = spotify.get_audio_features(track_id)

print(f"Danceability: {audio_features['danceability']}")
print(f"Energy: {audio_features['energy']}")
print(f"Valence: {audio_features['valence']}")
```

### YouTube Integration

```python
from src.integrations import YouTubeIntegration

youtube = YouTubeIntegration()

# Search music videos
videos = youtube.search_music('Pink Floyd - Comfortably Numb', max_results=5)

# Get video info
video_id = videos[0]['video_id']
video_info = youtube.get_video_info(video_id)

print(f"Title: {video_info['title']}")
print(f"Views: {video_info['view_count']:,}")
```

### Apple Music Integration

```python
from src.integrations import AppleMusicIntegration

apple_music = AppleMusicIntegration()

# Search songs
songs = apple_music.search_songs('Stairway to Heaven', limit=5)

# Get song info
song_id = songs[0]['id']
song_info = apple_music.get_song(song_id)

print(f"Artist: {song_info['artist']}")
print(f"Album: {song_info['album']}")
```

---

## 📊 Output Format

### Analysis Output Structure

```json
{
  "file_path": "music/song.mp3",
  "analysis_timestamp": "2025-10-15T10:30:00",
  "status": "success",
  "audio_features": {
    "bpm": 120.5,
    "estimated_key": "C",
    "duration_minutes": 3.5,
    "spectral_centroid_mean": 2341.5,
    "rms_mean": 0.123,
    "mfcc_1_mean": -123.45,
    ...
  },
  "genre": {
    "predicted_genre": "rock",
    "confidence": 0.87,
    "top_3_predictions": [
      {"genre": "rock", "probability": 0.87},
      {"genre": "pop", "probability": 0.08},
      {"genre": "metal", "probability": 0.03}
    ]
  },
  "mood": {
    "predicted_mood": "energetic",
    "confidence": 0.92,
    "valence_arousal": {
      "valence": 0.75,
      "arousal": 0.85,
      "valence_label": "positive",
      "arousal_label": "high"
    },
    "emotional_quadrant": "High Energy Positive (Happy/Excited)"
  },
  "api_data": {
    "spotify": { ... },
    "youtube": { ... },
    "apple_music": { ... }
  },
  "summary": "..."
}
```

---

## 🔑 API Setup

### Spotify API

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Copy **Client ID** and **Client Secret**
4. Add to `.env` file

### YouTube API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable **YouTube Data API v3**
4. Create credentials (API Key)
5. Add to `.env` file

### Apple Music API

1. Join [Apple Developer Program](https://developer.apple.com/programs/)
2. Create a MusicKit identifier
3. Generate a private key (.p8 file)
4. Get Key ID and Team ID
5. Add to `.env` file

---

## 🏗️ Project Structure

```
audio-analyzer/
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   └── audio_features.py       # Feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── genre_classifier.py     # Genre classification
│   │   └── mood_analyzer.py        # Mood analysis
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── spotify_integration.py  # Spotify API
│   │   ├── youtube_integration.py  # YouTube API
│   │   └── apple_music_integration.py  # Apple Music API
│   ├── utils/                      # Utility functions
│   ├── __init__.py
│   └── analyzer.py                 # Main pipeline
├── data/
│   ├── raw/                        # Raw audio files
│   ├── processed/                  # Processed data
│   └── models/                     # Trained models
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks
├── config/                         # Configuration files
├── .env.example                    # Example environment file
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🎓 Technical Details

### Audio Features (68 features total)
- **Temporal**: 7 features (BPM, beat count, onset strength, etc.)
- **Harmonic**: 7 features (key, chroma features, harmonic ratio)
- **Spectral**: 12 features (centroid, rolloff, bandwidth, contrast, flatness, ZCR)
- **MFCC**: 40 features (20 coefficients × mean/std)
- **Energy**: 7 features (RMS, loudness, dynamic range)
- **Duration**: 3 features

### Genre Classification Model
- Architecture: Dense(512) → BN → Dropout(0.3) → Dense(256) → BN → Dropout(0.3) → Dense(128) → BN → Dropout(0.2) → Dense(64) → BN → Dropout(0.2) → Dense(10, softmax)
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau

### Mood Analysis Model
- Algorithm: Random Forest (200 estimators, max_depth=20)
- Framework: Valence-Arousal (Russell's circumplex model)
- Features: 46 mood-relevant features
- Output: 8 mood categories + valence/arousal scores

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Hasan Arthur Altuntaş**

- GitHub: [@Rtur2003](https://github.com/Rtur2003)
- LinkedIn: [Hasan Arthur Altuntaş](https://www.linkedin.com/in/hasan-arthur-altuntas)
- Website: [hasanarthuraltuntas.xyz](https://hasanarthuraltuntas.xyz)

---

## 🙏 Acknowledgments

- [Librosa](https://librosa.org/) - Audio analysis library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library
- [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
- [YouTube Data API](https://developers.google.com/youtube/v3)
- [Apple Music API](https://developer.apple.com/documentation/applemusicapi)

---

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on [GitHub](https://github.com/Rtur2003/audio-analyzer/issues)
- Email: [Your email if you want to include it]

---

## 🚀 Future Improvements

- [ ] Add web interface (Flask/FastAPI)
- [ ] Real-time audio streaming analysis
- [ ] Playlist recommendation system
- [ ] Audio similarity search
- [ ] Beat-synchronized visualizations
- [ ] Model fine-tuning interface
- [ ] Support for more audio formats
- [ ] Docker containerization
- [ ] REST API endpoints

---

**⭐ If you find this project useful, please consider giving it a star!**
