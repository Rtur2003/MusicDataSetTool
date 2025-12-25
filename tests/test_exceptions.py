"""Tests for exception classes."""

from src.utils.exceptions import (
    MusicAnalyzerError,
    AudioLoadError,
    FeatureExtractionError,
    ModelError,
    APIError,
    ValidationError
)


def test_music_analyzer_error():
    """Test MusicAnalyzerError exception."""
    error = MusicAnalyzerError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, Exception)


def test_audio_load_error():
    """Test AudioLoadError exception."""
    error = AudioLoadError("Failed to load audio")
    assert str(error) == "Failed to load audio"
    assert isinstance(error, MusicAnalyzerError)


def test_feature_extraction_error():
    """Test FeatureExtractionError exception."""
    error = FeatureExtractionError("Feature extraction failed")
    assert str(error) == "Feature extraction failed"
    assert isinstance(error, MusicAnalyzerError)


def test_model_error():
    """Test ModelError exception."""
    error = ModelError("Model error")
    assert str(error) == "Model error"
    assert isinstance(error, MusicAnalyzerError)


def test_api_error():
    """Test APIError exception."""
    error = APIError("API request failed")
    assert str(error) == "API request failed"
    assert isinstance(error, MusicAnalyzerError)


def test_validation_error():
    """Test ValidationError exception."""
    error = ValidationError("Validation failed")
    assert str(error) == "Validation failed"
    assert isinstance(error, MusicAnalyzerError)
