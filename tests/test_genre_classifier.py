"""Tests for genre classifier."""

import pytest
import numpy as np
from src.models.genre_classifier import GenreClassifier
from src.utils.exceptions import ModelError, ValidationError


def test_genre_classifier_init():
    """Test GenreClassifier initialization."""
    classifier = GenreClassifier()
    assert classifier.model is None
    assert not classifier.is_trained
    assert len(classifier.GENRES) == 10


def test_prepare_features(sample_features, sample_mfcc_features):
    """Test feature preparation."""
    classifier = GenreClassifier()
    features = {**sample_features, **sample_mfcc_features}
    result = classifier.prepare_features(features)
    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_rule_based_prediction(sample_features, sample_mfcc_features):
    """Test rule-based prediction when model not trained."""
    classifier = GenreClassifier()
    features = {**sample_features, **sample_mfcc_features}
    result = classifier.predict(features)
    
    assert 'predicted_genre' in result
    assert 'confidence' in result
    assert 'method' in result
    assert result['method'] == 'rule_based'


def test_predict_empty_features():
    """Test prediction with empty features."""
    classifier = GenreClassifier()
    with pytest.raises(ValidationError, match="cannot be empty"):
        classifier.predict({})


def test_save_model_not_trained(tmp_path):
    """Test saving untrained model."""
    classifier = GenreClassifier()
    with pytest.raises(ModelError, match="must be trained"):
        classifier.save_model(str(tmp_path))


def test_load_model_not_found(tmp_path):
    """Test loading non-existent model."""
    classifier = GenreClassifier()
    with pytest.raises(ModelError, match="not found"):
        classifier.load_model(str(tmp_path))
