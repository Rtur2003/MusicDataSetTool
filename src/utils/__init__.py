"""Utility modules for the Music Dataset Tool."""

from .logger import get_logger
from .validators import validate_file_path, validate_url, validate_positive_int
from .exceptions import (
    MusicAnalyzerError,
    AudioLoadError,
    FeatureExtractionError,
    ModelError,
    APIError,
    ValidationError
)
from .retry import retry_with_backoff, retry_on_rate_limit
from .cache import SimpleCache, cached

__all__ = [
    'get_logger',
    'validate_file_path',
    'validate_url',
    'validate_positive_int',
    'MusicAnalyzerError',
    'AudioLoadError',
    'FeatureExtractionError',
    'ModelError',
    'APIError',
    'ValidationError',
    'retry_with_backoff',
    'retry_on_rate_limit',
    'SimpleCache',
    'cached'
]
