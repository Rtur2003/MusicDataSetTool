"""Input validation utilities."""

import os
from pathlib import Path
from typing import Union
from urllib.parse import urlparse
from .exceptions import ValidationError


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate a file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If validation fails
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")
        
    path = Path(file_path)
    
    if must_exist:
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
            
    return path


def validate_url(url: str) -> str:
    """
    Validate a URL.
    
    Args:
        url: URL to validate
        
    Returns:
        Validated URL string
        
    Raises:
        ValidationError: If validation fails
    """
    if not url:
        raise ValidationError("URL cannot be empty")
        
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValidationError(f"Invalid URL format: {url}")
    except Exception as e:
        raise ValidationError(f"Invalid URL: {url} - {str(e)}")
        
    return url


def validate_positive_int(value: int, name: str = "value", min_value: int = 1) -> int:
    """
    Validate a positive integer.
    
    Args:
        value: Integer to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value
        
    Returns:
        Validated integer
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
        
    if value < min_value:
        raise ValidationError(f"{name} must be at least {min_value}, got {value}")
        
    return value


def validate_audio_format(file_path: Path) -> bool:
    """
    Check if file has a supported audio format.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if format is supported
        
    Raises:
        ValidationError: If format is not supported
    """
    supported_formats = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    suffix = file_path.suffix.lower()
    
    if suffix not in supported_formats:
        raise ValidationError(
            f"Unsupported audio format: {suffix}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
        
    return True
