"""Custom exceptions for the Music Dataset Tool."""


class MusicAnalyzerError(Exception):
    """Base exception for Music Analyzer errors."""
    pass


class AudioLoadError(MusicAnalyzerError):
    """Raised when audio file cannot be loaded."""
    pass


class FeatureExtractionError(MusicAnalyzerError):
    """Raised when feature extraction fails."""
    pass


class ModelError(MusicAnalyzerError):
    """Raised when model operations fail."""
    pass


class APIError(MusicAnalyzerError):
    """Raised when API requests fail."""
    pass


class ValidationError(MusicAnalyzerError):
    """Raised when input validation fails."""
    pass
