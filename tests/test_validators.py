"""Tests for validation utilities."""

import pytest
from pathlib import Path
from src.utils.validators import (
    validate_file_path,
    validate_url,
    validate_positive_int,
    validate_audio_format
)
from src.utils.exceptions import ValidationError


def test_validate_file_path_exists(temp_audio_file):
    """Test file path validation with existing file."""
    result = validate_file_path(temp_audio_file, must_exist=True)
    assert isinstance(result, Path)
    assert result.exists()


def test_validate_file_path_not_exists():
    """Test file path validation with non-existing file."""
    with pytest.raises(ValidationError, match="File does not exist"):
        validate_file_path("/nonexistent/file.mp3", must_exist=True)


def test_validate_file_path_empty():
    """Test file path validation with empty path."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_file_path("", must_exist=False)


def test_validate_url_valid():
    """Test URL validation with valid URL."""
    url = "https://example.com/test"
    result = validate_url(url)
    assert result == url


def test_validate_url_invalid():
    """Test URL validation with invalid URL."""
    with pytest.raises(ValidationError, match="Invalid URL"):
        validate_url("not a url")


def test_validate_url_empty():
    """Test URL validation with empty URL."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_url("")


def test_validate_positive_int_valid():
    """Test positive integer validation with valid value."""
    result = validate_positive_int(5, "test_param", min_value=1)
    assert result == 5


def test_validate_positive_int_negative():
    """Test positive integer validation with negative value."""
    with pytest.raises(ValidationError, match="must be at least"):
        validate_positive_int(-1, "test_param", min_value=1)


def test_validate_positive_int_zero():
    """Test positive integer validation with zero."""
    with pytest.raises(ValidationError, match="must be at least"):
        validate_positive_int(0, "test_param", min_value=1)


def test_validate_positive_int_not_int():
    """Test positive integer validation with non-integer."""
    with pytest.raises(ValidationError, match="must be an integer"):
        validate_positive_int(1.5, "test_param")


def test_validate_audio_format_valid(tmp_path):
    """Test audio format validation with valid format."""
    audio_file = tmp_path / "test.mp3"
    assert validate_audio_format(audio_file) is True


def test_validate_audio_format_invalid(tmp_path):
    """Test audio format validation with invalid format."""
    invalid_file = tmp_path / "test.txt"
    with pytest.raises(ValidationError, match="Unsupported audio format"):
        validate_audio_format(invalid_file)
