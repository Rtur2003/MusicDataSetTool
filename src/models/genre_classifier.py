"""
Music Genre Classification Model
Uses TensorFlow/Keras for classifying music into different genres
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
import os

from ..utils.logger import get_logger
from ..utils.exceptions import ModelError, ValidationError
from ..utils.constants import (
    GENRES, NEURAL_NETWORK_LAYERS, DROPOUT_RATES, LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LEARNING_RATE
)

logger = get_logger(__name__)


class GenreClassifier:
    """Deep learning model for music genre classification"""

    GENRES = GENRES

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the genre classifier

        Args:
            model_path: Path to load a pre-trained model
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.GENRES)
        self.is_trained = False
        
        logger.info("GenreClassifier initialized")

        if model_path and os.path.exists(model_path):
            try:
                self.load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {str(e)}")

    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build a neural network model for genre classification

        Args:
            input_dim: Number of input features

        Returns:
            Compiled Keras model
            
        Raises:
            ModelError: If model building fails
        """
        try:
            layers_config = NEURAL_NETWORK_LAYERS
            dropout_config = DROPOUT_RATES
            
            model = models.Sequential([
                layers.Dense(layers_config[0], activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(dropout_config[0]),

                layers.Dense(layers_config[1], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_config[1]),

                layers.Dense(layers_config[2], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_config[2]),

                layers.Dense(layers_config[3], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_config[3]),

                layers.Dense(len(self.GENRES), activation='softmax')
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Model built with {input_dim} input features")
            return model
            
        except Exception as e:
            logger.error(f"Failed to build model: {str(e)}")
            raise ModelError(f"Model building failed: {str(e)}")

    def prepare_features(self, features: Dict) -> np.ndarray:
        """
        Prepare features for model input

        Args:
            features: Dictionary of audio features

        Returns:
            Numpy array of prepared features
        """
        # Select relevant features for classification
        feature_keys = [
            'bpm', 'beat_count', 'beats_per_second',
            'onset_strength_mean', 'onset_strength_std',
            'key_strength', 'chroma_stft_mean', 'chroma_stft_std',
            'chroma_cqt_mean', 'chroma_cens_mean', 'harmonic_ratio',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_contrast_mean', 'spectral_contrast_std',
            'spectral_flatness_mean', 'spectral_flatness_std',
            'zero_crossing_rate_mean', 'zero_crossing_rate_std',
            'rms_mean', 'rms_std', 'rms_max',
            'loudness_mean', 'loudness_std', 'dynamic_range'
        ]

        # Add MFCC features
        for i in range(1, 21):
            feature_keys.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])

        # Extract feature values
        feature_values = []
        for key in feature_keys:
            value = features.get(key, 0.0)
            feature_values.append(float(value))

        return np.array(feature_values)

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32) -> Dict:
        """
        Train the genre classification model

        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            validation_split: Validation data split ratio
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training history dictionary
            
        Raises:
            ValidationError: If input validation fails
            ModelError: If training fails
        """
        if X.shape[0] != y.shape[0]:
            raise ValidationError(f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}")
            
        if X.shape[0] < 10:
            raise ValidationError("Need at least 10 samples for training")
            
        if not 0 < validation_split < 1:
            raise ValidationError(f"validation_split must be between 0 and 1, got {validation_split}")
            
        try:
            logger.info(f"Training genre classifier with {X.shape[0]} samples, {X.shape[1]} features")
            
            X_scaled = self.scaler.fit_transform(X)
            y_encoded = self.label_encoder.transform(y)

            self.model = self.build_model(X.shape[1])

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=REDUCE_LR_FACTOR,
                    patience=REDUCE_LR_PATIENCE,
                    min_lr=MIN_LEARNING_RATE
                )
            ]

            history = self.model.fit(
                X_scaled, y_encoded,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            self.is_trained = True
            logger.info("Training completed successfully")
            
            return history.history
            
        except (ValidationError, ModelError) as e:
            raise
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise ModelError(f"Training failed: {str(e)}")

    def predict(self, features: Dict) -> Dict:
        """
        Predict genre for a single audio file

        Args:
            features: Dictionary of audio features

        Returns:
            Dictionary with predicted genre and probabilities
            
        Raises:
            ValidationError: If features are invalid
            ModelError: If prediction fails
        """
        if not features:
            raise ValidationError("Features dictionary cannot be empty")
            
        if not self.is_trained and self.model is None:
            logger.info("Model not trained, using rule-based prediction")
            return self._get_rule_based_prediction(features)

        try:
            X = self.prepare_features(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            predictions = self.model.predict(X_scaled, verbose=0)[0]

            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_genres = [self.GENRES[i] for i in top_3_indices]
            top_3_probs = [float(predictions[i]) for i in top_3_indices]

            return {
                'predicted_genre': top_3_genres[0],
                'confidence': top_3_probs[0],
                'top_3_predictions': [
                    {'genre': genre, 'probability': prob}
                    for genre, prob in zip(top_3_genres, top_3_probs)
                ],
                'all_probabilities': {
                    genre: float(predictions[i])
                    for i, genre in enumerate(self.GENRES)
                }
            }
        except (ValidationError, ModelError) as e:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}")

    def _get_rule_based_prediction(self, features: Dict) -> Dict:
        """
        Rule-based prediction when model is not trained
        Uses heuristics based on audio features

        Args:
            features: Dictionary of audio features

        Returns:
            Dictionary with predicted genre
        """
        # Simple rule-based classification
        bpm = features.get('bpm', 120)
        spectral_centroid = features.get('spectral_centroid_mean', 2000)
        energy = features.get('rms_mean', 0.1)
        harmonic_ratio = features.get('harmonic_ratio', 0.5)

        # Rule-based logic
        if bpm > 140 and energy > 0.15:
            genre = 'electronic'
        elif bpm > 120 and spectral_centroid > 3000:
            genre = 'rock'
        elif bpm < 80 and harmonic_ratio > 0.7:
            genre = 'classical'
        elif 80 <= bpm <= 100 and harmonic_ratio > 0.6:
            genre = 'jazz'
        elif bpm > 100 and energy > 0.12:
            genre = 'pop'
        else:
            genre = 'unknown'

        return {
            'predicted_genre': genre,
            'confidence': 0.5,
            'method': 'rule_based',
            'note': 'Model not trained. Using rule-based prediction.',
            'top_3_predictions': [
                {'genre': genre, 'probability': 0.5}
            ]
        }

    def save_model(self, model_dir: str, model_name: str = 'genre_classifier'):
        """
        Save the trained model and preprocessing objects

        Args:
            model_dir: Directory to save the model
            model_name: Base name for the model files
            
        Raises:
            ModelError: If model is not trained or save fails
        """
        if not self.is_trained:
            raise ModelError("Model must be trained before saving")

        try:
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)

            model_file = model_path / f'{model_name}.h5'
            self.model.save(str(model_file))

            scaler_file = model_path / f'{model_name}_scaler.joblib'
            joblib.dump(self.scaler, str(scaler_file))

            logger.info(f"Model saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise ModelError(f"Model save failed: {str(e)}")

    def load_model(self, model_dir: str, model_name: str = 'genre_classifier'):
        """
        Load a trained model and preprocessing objects

        Args:
            model_dir: Directory containing the model files
            model_name: Base name of the model files
            
        Raises:
            ModelError: If model files not found or load fails
        """
        try:
            model_path = Path(model_dir) / f'{model_name}.h5'
            if not model_path.exists():
                raise ModelError(f"Model not found at {model_path}")
                
            self.model = keras.models.load_model(str(model_path))
            self.is_trained = True

            scaler_path = Path(model_dir) / f'{model_name}_scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(str(scaler_path))
            else:
                logger.warning(f"Scaler not found at {scaler_path}")

            logger.info(f"Model loaded from {model_dir}")
            
        except ModelError as e:
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelError(f"Model load failed: {str(e)}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance

        Args:
            X: Feature array
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        X_scaled = self.scaler.transform(X)
        y_encoded = self.label_encoder.transform(y)

        loss, accuracy = self.model.evaluate(X_scaled, y_encoded, verbose=0)

        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }


# Example usage
if __name__ == "__main__":
    classifier = GenreClassifier()

    print("GenreClassifier initialized successfully!")
    print(f"Supported genres: {', '.join(classifier.GENRES)}")
    print("\nAvailable methods:")
    print("  - train(X, y)")
    print("  - predict(features)")
    print("  - save_model(model_dir)")
    print("  - load_model(model_dir)")
    print("  - evaluate(X, y)")
