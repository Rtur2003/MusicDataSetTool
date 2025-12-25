"""Tests for constants module."""

from src.utils.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_MFCC,
    GENRES,
    MOODS,
    VALENCE_AROUSAL_MAP
)


def test_default_sample_rate():
    """Test default sample rate constant."""
    assert DEFAULT_SAMPLE_RATE == 22050
    assert isinstance(DEFAULT_SAMPLE_RATE, int)


def test_default_n_mfcc():
    """Test default number of MFCCs constant."""
    assert DEFAULT_N_MFCC == 20
    assert isinstance(DEFAULT_N_MFCC, int)


def test_genres_list():
    """Test genres list constant."""
    assert isinstance(GENRES, list)
    assert len(GENRES) == 10
    assert 'rock' in GENRES
    assert 'pop' in GENRES
    assert 'jazz' in GENRES


def test_moods_list():
    """Test moods list constant."""
    assert isinstance(MOODS, list)
    assert len(MOODS) == 8
    assert 'happy' in MOODS
    assert 'sad' in MOODS
    assert 'energetic' in MOODS


def test_valence_arousal_map():
    """Test valence-arousal mapping."""
    assert isinstance(VALENCE_AROUSAL_MAP, dict)
    assert len(VALENCE_AROUSAL_MAP) == len(MOODS)
    
    for mood in MOODS:
        assert mood in VALENCE_AROUSAL_MAP
        assert 'valence' in VALENCE_AROUSAL_MAP[mood]
        assert 'arousal' in VALENCE_AROUSAL_MAP[mood]
        assert 0 <= VALENCE_AROUSAL_MAP[mood]['valence'] <= 1
        assert 0 <= VALENCE_AROUSAL_MAP[mood]['arousal'] <= 1
