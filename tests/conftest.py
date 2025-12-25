"""Test configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_features():
    """Sample audio features for testing."""
    return {
        'bpm': 120.0,
        'beat_count': 100,
        'beats_per_second': 2.0,
        'onset_strength_mean': 0.5,
        'onset_strength_std': 0.2,
        'key_strength': 0.7,
        'estimated_key': 'C',
        'harmonic_ratio': 0.6,
        'spectral_centroid_mean': 2000.0,
        'spectral_rolloff_mean': 3000.0,
        'rms_mean': 0.1,
        'loudness_mean': -20.0,
        'duration_seconds': 180.0,
        'duration_minutes': 3.0,
        'sample_rate': 22050
    }


@pytest.fixture
def sample_mfcc_features():
    """Sample MFCC features for testing."""
    features = {}
    for i in range(1, 21):
        features[f'mfcc_{i}_mean'] = float(np.random.randn())
        features[f'mfcc_{i}_std'] = float(np.abs(np.random.randn()))
    return features


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.touch()
    return audio_file
