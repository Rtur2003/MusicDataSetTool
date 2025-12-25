"""Application-wide constants."""

from typing import List, Dict

DEFAULT_SAMPLE_RATE: int = 22050
DEFAULT_N_MFCC: int = 20

SUPPORTED_AUDIO_FORMATS: List[str] = [
    '.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'
]

GENRES: List[str] = [
    'rock', 'pop', 'jazz', 'classical', 'hip-hop',
    'electronic', 'country', 'blues', 'reggae', 'metal'
]

MOODS: List[str] = [
    'happy', 'sad', 'energetic', 'calm',
    'angry', 'romantic', 'melancholic', 'uplifting'
]

VALENCE_AROUSAL_MAP: Dict[str, Dict[str, float]] = {
    'happy': {'valence': 0.8, 'arousal': 0.7},
    'sad': {'valence': 0.2, 'arousal': 0.3},
    'energetic': {'valence': 0.7, 'arousal': 0.9},
    'calm': {'valence': 0.6, 'arousal': 0.2},
    'angry': {'valence': 0.2, 'arousal': 0.8},
    'romantic': {'valence': 0.7, 'arousal': 0.4},
    'melancholic': {'valence': 0.3, 'arousal': 0.4},
    'uplifting': {'valence': 0.9, 'arousal': 0.6}
}

API_REQUEST_TIMEOUT: int = 30
API_MAX_RETRIES: int = 3
API_RETRY_DELAY: int = 1

MODEL_SAVE_FORMAT: str = 'h5'
SCALER_SAVE_FORMAT: str = 'joblib'

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 100
DEFAULT_VALIDATION_SPLIT: float = 0.2

RANDOM_FOREST_N_ESTIMATORS: int = 200
RANDOM_FOREST_MAX_DEPTH: int = 20
RANDOM_FOREST_MIN_SAMPLES_SPLIT: int = 5
RANDOM_FOREST_MIN_SAMPLES_LEAF: int = 2

NEURAL_NETWORK_LAYERS: List[int] = [512, 256, 128, 64]
DROPOUT_RATES: List[float] = [0.3, 0.3, 0.2, 0.2]
LEARNING_RATE: float = 0.001

EARLY_STOPPING_PATIENCE: int = 15
REDUCE_LR_PATIENCE: int = 5
REDUCE_LR_FACTOR: float = 0.5
MIN_LEARNING_RATE: float = 1e-6
