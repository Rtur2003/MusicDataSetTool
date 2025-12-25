"""Configuration management for the Music Dataset Tool."""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = int(os.getenv('AUDIO_SAMPLE_RATE', '22050'))
    n_mfcc: int = 20
    hop_length: int = 512
    n_fft: int = 2048


@dataclass
class ModelConfig:
    """Model training configuration."""
    model_dir: Path = Path(os.getenv('MODEL_PATH', 'data/models/'))
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5


@dataclass
class SpotifyConfig:
    """Spotify API configuration."""
    client_id: Optional[str] = field(default_factory=lambda: os.getenv('SPOTIFY_CLIENT_ID'))
    client_secret: Optional[str] = field(default_factory=lambda: os.getenv('SPOTIFY_CLIENT_SECRET'))
    
    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


@dataclass
class YouTubeConfig:
    """YouTube API configuration."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv('YOUTUBE_API_KEY'))
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class AppleMusicConfig:
    """Apple Music API configuration."""
    key_id: Optional[str] = field(default_factory=lambda: os.getenv('APPLE_MUSIC_KEY_ID'))
    team_id: Optional[str] = field(default_factory=lambda: os.getenv('APPLE_MUSIC_TEAM_ID'))
    private_key_path: Optional[str] = field(
        default_factory=lambda: os.getenv('APPLE_MUSIC_PRIVATE_KEY_PATH')
    )
    
    @property
    def is_configured(self) -> bool:
        return bool(self.key_id and self.team_id and self.private_key_path)


@dataclass
class AppConfig:
    """Main application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    spotify: SpotifyConfig = field(default_factory=SpotifyConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    apple_music: AppleMusicConfig = field(default_factory=AppleMusicConfig)
    
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_file: Optional[str] = os.getenv('LOG_FILE')
    
    def __post_init__(self):
        """Ensure required directories exist."""
        self.model.model_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    """Get the application configuration."""
    return AppConfig()
