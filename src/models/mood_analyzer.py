"""
Music Mood/Emotion Analyzer
Uses scikit-learn for classifying music mood and emotional characteristics
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
import os

from ..utils.logger import get_logger
from ..utils.exceptions import ModelError, ValidationError
from ..utils.constants import (
    MOODS, VALENCE_AROUSAL_MAP,
    RANDOM_FOREST_N_ESTIMATORS, RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MIN_SAMPLES_SPLIT, RANDOM_FOREST_MIN_SAMPLES_LEAF
)

logger = get_logger(__name__)


class MoodAnalyzer:
    """Machine learning model for music mood/emotion analysis"""

    MOODS = MOODS
    VALENCE_AROUSAL_MAP = VALENCE_AROUSAL_MAP

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the mood analyzer

        Args:
            model_path: Path to load a pre-trained model
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.MOODS)
        self.is_trained = False
        
        logger.info("MoodAnalyzer initialized")

        if model_path and os.path.exists(model_path):
            try:
                self.load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {str(e)}")

    def build_model(self) -> RandomForestClassifier:
        """
        Build a Random Forest classifier for mood analysis

        Returns:
            RandomForestClassifier instance
            
        Raises:
            ModelError: If model building fails
        """
        try:
            model = RandomForestClassifier(
                n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                max_depth=RANDOM_FOREST_MAX_DEPTH,
                min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT,
                min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
                random_state=42,
                n_jobs=-1
            )
            logger.info("Random Forest model built successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to build model: {str(e)}")
            raise ModelError(f"Model building failed: {str(e)}")

    def prepare_features(self, features: Dict) -> np.ndarray:
        """
        Prepare features for mood analysis

        Args:
            features: Dictionary of audio features

        Returns:
            Numpy array of prepared features
        """
        # Select mood-relevant features
        feature_keys = [
            'bpm', 'beats_per_second',
            'onset_strength_mean', 'onset_strength_std',
            'key_strength', 'harmonic_ratio',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_bandwidth_mean',
            'spectral_contrast_mean', 'spectral_flatness_mean',
            'zero_crossing_rate_mean',
            'rms_mean', 'rms_std', 'rms_max',
            'loudness_mean', 'loudness_std', 'dynamic_range',
            'chroma_stft_mean', 'chroma_cqt_mean'
        ]

        # Add first 13 MFCC features (most relevant for emotion)
        for i in range(1, 14):
            feature_keys.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])

        # Extract feature values
        feature_values = []
        for key in feature_keys:
            value = features.get(key, 0.0)
            feature_values.append(float(value))

        return np.array(feature_values)

    def calculate_valence_arousal(self, features: Dict) -> Dict:
        """
        Calculate valence (positive/negative) and arousal (energy) values

        Args:
            features: Dictionary of audio features

        Returns:
            Dictionary with valence and arousal scores
        """
        # Valence indicators (positive/negative emotion)
        harmonic_ratio = features.get('harmonic_ratio', 0.5)
        key_strength = features.get('key_strength', 0.5)
        spectral_centroid = features.get('spectral_centroid_mean', 2000)

        # Normalize and calculate valence (0-1 scale)
        valence = (
            harmonic_ratio * 0.4 +
            key_strength * 0.3 +
            min(spectral_centroid / 4000, 1.0) * 0.3
        )

        # Arousal indicators (energy/excitement)
        bpm = features.get('bpm', 120)
        rms_energy = features.get('rms_mean', 0.1)
        loudness = features.get('loudness_mean', -20)

        # Normalize and calculate arousal (0-1 scale)
        arousal = (
            min(bpm / 180, 1.0) * 0.4 +
            min(rms_energy / 0.3, 1.0) * 0.3 +
            min((loudness + 60) / 60, 1.0) * 0.3
        )

        return {
            'valence': float(np.clip(valence, 0, 1)),
            'arousal': float(np.clip(arousal, 0, 1)),
            'valence_label': 'positive' if valence > 0.5 else 'negative',
            'arousal_label': 'high' if arousal > 0.5 else 'low'
        }

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2) -> Dict:
        """
        Train the mood analysis model

        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            validation_split: Validation data split ratio

        Returns:
            Training results dictionary
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels
        y_encoded = self.label_encoder.transform(y)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded,
            test_size=validation_split,
            random_state=42,
            stratify=y_encoded
        )

        # Build and train model
        self.model = self.build_model()
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        self.is_trained = True

        return {
            'train_accuracy': float(train_score),
            'val_accuracy': float(val_score),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

    def predict(self, features: Dict) -> Dict:
        """
        Predict mood for a single audio file

        Args:
            features: Dictionary of audio features

        Returns:
            Dictionary with predicted mood and analysis
        """
        # Calculate valence and arousal
        valence_arousal = self.calculate_valence_arousal(features)

        # If model is trained, use it
        if self.is_trained and self.model is not None:
            # Prepare features
            X = self.prepare_features(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Predict probabilities
            probabilities = self.model.predict_proba(X_scaled)[0]

            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_moods = [self.MOODS[i] for i in top_3_indices]
            top_3_probs = [float(probabilities[i]) for i in top_3_indices]

            result = {
                'predicted_mood': top_3_moods[0],
                'confidence': top_3_probs[0],
                'top_3_predictions': [
                    {'mood': mood, 'probability': prob}
                    for mood, prob in zip(top_3_moods, top_3_probs)
                ],
                'all_probabilities': {
                    mood: float(probabilities[i])
                    for i, mood in enumerate(self.MOODS)
                }
            }
        else:
            # Use rule-based prediction
            result = self._get_rule_based_prediction(features, valence_arousal)

        # Add valence/arousal analysis
        result.update({
            'valence_arousal': valence_arousal,
            'emotional_quadrant': self._get_emotional_quadrant(valence_arousal)
        })

        return result

    def _get_rule_based_prediction(self, features: Dict,
                                   valence_arousal: Dict) -> Dict:
        """
        Rule-based mood prediction using valence-arousal model

        Args:
            features: Dictionary of audio features
            valence_arousal: Valence and arousal scores

        Returns:
            Dictionary with predicted mood
        """
        valence = valence_arousal['valence']
        arousal = valence_arousal['arousal']

        # Map valence-arousal to mood
        if valence > 0.6 and arousal > 0.6:
            mood = 'energetic'
        elif valence > 0.6 and arousal < 0.4:
            mood = 'calm'
        elif valence < 0.4 and arousal > 0.6:
            mood = 'angry'
        elif valence < 0.4 and arousal < 0.4:
            mood = 'sad'
        elif valence > 0.7:
            mood = 'happy'
        elif valence < 0.3:
            mood = 'melancholic'
        else:
            mood = 'romantic'

        return {
            'predicted_mood': mood,
            'confidence': 0.6,
            'method': 'rule_based',
            'note': 'Model not trained. Using valence-arousal mapping.',
            'top_3_predictions': [
                {'mood': mood, 'probability': 0.6}
            ]
        }

    def _get_emotional_quadrant(self, valence_arousal: Dict) -> str:
        """
        Determine emotional quadrant based on valence-arousal

        Args:
            valence_arousal: Valence and arousal scores

        Returns:
            Emotional quadrant description
        """
        valence = valence_arousal['valence']
        arousal = valence_arousal['arousal']

        if valence > 0.5 and arousal > 0.5:
            return 'High Energy Positive (Happy/Excited)'
        elif valence > 0.5 and arousal <= 0.5:
            return 'Low Energy Positive (Calm/Relaxed)'
        elif valence <= 0.5 and arousal > 0.5:
            return 'High Energy Negative (Angry/Tense)'
        else:
            return 'Low Energy Negative (Sad/Depressed)'

    def get_mood_description(self, mood: str) -> str:
        """
        Get a description of the mood

        Args:
            mood: Mood label

        Returns:
            Description string
        """
        descriptions = {
            'happy': 'Cheerful and upbeat music that evokes joy and positivity',
            'sad': 'Melancholic and somber music that evokes sadness or reflection',
            'energetic': 'High-energy music with strong rhythm and excitement',
            'calm': 'Peaceful and relaxing music with low energy',
            'angry': 'Aggressive and intense music with strong emotions',
            'romantic': 'Tender and affectionate music that evokes love',
            'melancholic': 'Bittersweet and nostalgic music with emotional depth',
            'uplifting': 'Inspirational and motivating music that raises spirits'
        }
        return descriptions.get(mood, 'Unknown mood')

    def save_model(self, model_dir: str, model_name: str = 'mood_analyzer'):
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

            model_file = model_path / f'{model_name}.joblib'
            joblib.dump(self.model, str(model_file))

            scaler_file = model_path / f'{model_name}_scaler.joblib'
            joblib.dump(self.scaler, str(scaler_file))

            logger.info(f"Model saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise ModelError(f"Model save failed: {str(e)}")

    def load_model(self, model_dir: str, model_name: str = 'mood_analyzer'):
        """
        Load a trained model and preprocessing objects

        Args:
            model_dir: Directory containing the model files
            model_name: Base name of the model files
            
        Raises:
            ModelError: If model files not found or load fails
        """
        try:
            model_path = Path(model_dir) / f'{model_name}.joblib'
            if not model_path.exists():
                raise ModelError(f"Model not found at {model_path}")
                
            self.model = joblib.load(str(model_path))
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

        accuracy = self.model.score(X_scaled, y_encoded)

        return {
            'accuracy': float(accuracy),
            'n_samples': len(X)
        }


# Example usage
if __name__ == "__main__":
    analyzer = MoodAnalyzer()

    print("MoodAnalyzer initialized successfully!")
    print(f"Supported moods: {', '.join(analyzer.MOODS)}")
    print("\nValence-Arousal Emotion Model:")
    print("  Valence: Positive/Negative emotion (0-1)")
    print("  Arousal: Energy/Excitement level (0-1)")
    print("\nAvailable methods:")
    print("  - train(X, y)")
    print("  - predict(features)")
    print("  - calculate_valence_arousal(features)")
    print("  - save_model(model_dir)")
    print("  - load_model(model_dir)")
