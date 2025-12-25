"""
Audio Feature Extraction Module
Extracts comprehensive audio features including BPM, key, tempo, chroma, spectral features, etc.
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import get_logger
from ..utils.exceptions import AudioLoadError, FeatureExtractionError
from ..utils.validators import validate_file_path, validate_audio_format
from ..utils.constants import DEFAULT_SAMPLE_RATE, DEFAULT_N_MFCC

logger = get_logger(__name__)


class AudioFeatureExtractor:
    """Extract comprehensive audio features from music files"""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, n_mfcc: int = DEFAULT_N_MFCC):
        """
        Initialize the audio feature extractor

        Args:
            sample_rate: Target sample rate for audio processing
            n_mfcc: Number of MFCCs to extract
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        logger.info(f"Initialized AudioFeatureExtractor (sr={sample_rate}, n_mfcc={n_mfcc})")

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioLoadError: If audio file cannot be loaded
        """
        try:
            path = validate_file_path(file_path, must_exist=True)
            validate_audio_format(path)
            
            logger.debug(f"Loading audio file: {path}")
            y, sr = librosa.load(str(path), sr=self.sample_rate)
            
            if y is None or len(y) == 0:
                raise AudioLoadError(f"Loaded audio is empty: {file_path}")
                
            logger.debug(f"Audio loaded successfully: {len(y)} samples, {sr} Hz")
            return y, sr
            
        except (AudioLoadError, Exception) as e:
            logger.error(f"Failed to load audio file {file_path}: {str(e)}")
            raise AudioLoadError(f"Error loading audio file {file_path}: {str(e)}")

    def extract_tempo_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Extract tempo and rhythm-related features

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary with tempo features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            
            duration_seconds = len(y) / sr
            beats_per_second = len(beats) / duration_seconds if duration_seconds > 0 else 0

            return {
                'bpm': float(tempo),
                'beat_count': int(len(beats)),
                'beats_per_second': float(beats_per_second),
                'onset_strength_mean': float(np.mean(onset_env)),
                'onset_strength_std': float(np.std(onset_env)),
                'tempogram_mean': float(np.mean(tempogram)),
                'tempogram_std': float(np.std(tempogram))
            }
        except Exception as e:
            logger.error(f"Failed to extract tempo features: {str(e)}")
            raise FeatureExtractionError(f"Error extracting tempo features: {str(e)}")

    def extract_key_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Extract key and harmonic features

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary with key features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

            chroma_vals = np.mean(chroma, axis=1)
            key_index = np.argmax(chroma_vals)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            estimated_key = keys[key_index]

            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            total_energy = np.sum(y**2)
            harmonic_ratio = float(np.sum(y_harmonic**2) / total_energy) if total_energy > 0 else 0.0

            return {
                'estimated_key': estimated_key,
                'key_strength': float(chroma_vals[key_index]),
                'chroma_stft_mean': float(np.mean(chroma)),
                'chroma_stft_std': float(np.std(chroma)),
                'chroma_cqt_mean': float(np.mean(chroma_cqt)),
                'chroma_cens_mean': float(np.mean(chroma_cens)),
                'harmonic_ratio': harmonic_ratio
            }
        except Exception as e:
            logger.error(f"Failed to extract key features: {str(e)}")
            raise FeatureExtractionError(f"Error extracting key features: {str(e)}")

    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Extract spectral features

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary with spectral features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]

            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'spectral_contrast_mean': float(np.mean(spectral_contrast)),
                'spectral_contrast_std': float(np.std(spectral_contrast)),
                'spectral_flatness_mean': float(np.mean(spectral_flatness)),
                'spectral_flatness_std': float(np.std(spectral_flatness)),
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'zero_crossing_rate_std': float(np.std(zcr))
            }
        except Exception as e:
            logger.error(f"Failed to extract spectral features: {str(e)}")
            raise FeatureExtractionError(f"Error extracting spectral features: {str(e)}")

    def extract_mfcc_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Extract MFCC (Mel-frequency cepstral coefficients) features

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary with MFCC features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

            mfcc_features = {}
            for i in range(self.n_mfcc):
                mfcc_features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                mfcc_features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))

            return mfcc_features
        except Exception as e:
            logger.error(f"Failed to extract MFCC features: {str(e)}")
            raise FeatureExtractionError(f"Error extracting MFCC features: {str(e)}")

    def extract_energy_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Extract energy-related features

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary with energy features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            rms = librosa.feature.rms(y=y)[0]
            S = np.abs(librosa.stft(y))
            power = np.sum(S**2, axis=0)
            
            power = np.maximum(power, 1e-10)
            
            loudness = librosa.power_to_db(power)
            
            rms_min = np.min(rms)
            rms_max = np.max(rms)
            dynamic_range = float(rms_max - rms_min)

            return {
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms)),
                'rms_max': float(rms_max),
                'loudness_mean': float(np.mean(loudness)),
                'loudness_std': float(np.std(loudness)),
                'loudness_max': float(np.max(loudness)),
                'dynamic_range': dynamic_range
            }
        except Exception as e:
            logger.error(f"Failed to extract energy features: {str(e)}")
            raise FeatureExtractionError(f"Error extracting energy features: {str(e)}")

    def extract_duration_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Extract duration and time-related features

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary with duration features
        """
        duration = librosa.get_duration(y=y, sr=sr)

        return {
            'duration_seconds': float(duration),
            'duration_minutes': float(duration / 60),
            'total_samples': len(y)
        }

    def extract_all_features(self, file_path: str) -> Dict:
        """
        Extract all audio features from a file

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary containing all extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            logger.info(f"Extracting features from: {file_path}")
            
            y, sr = self.load_audio(file_path)

            features = {
                'file_path': file_path,
                'sample_rate': sr
            }

            features.update(self.extract_duration_features(y, sr))
            features.update(self.extract_tempo_features(y, sr))
            features.update(self.extract_key_features(y, sr))
            features.update(self.extract_spectral_features(y, sr))
            features.update(self.extract_mfcc_features(y, sr))
            features.update(self.extract_energy_features(y, sr))
            
            logger.info(f"Successfully extracted {len(features)} features")
            return features

        except (AudioLoadError, FeatureExtractionError) as e:
            raise
        except Exception as e:
            logger.error(f"Unexpected error extracting features: {str(e)}")
            raise FeatureExtractionError(f"Error extracting features from {file_path}: {str(e)}")

    def get_feature_summary(self, features: Dict) -> str:
        """
        Create a human-readable summary of features

        Args:
            features: Dictionary of extracted features

        Returns:
            Formatted string summary
        """
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              AUDIO FEATURE ANALYSIS SUMMARY                  ║
╠══════════════════════════════════════════════════════════════╣
║ File: {features.get('file_path', 'Unknown'):<52} ║
║                                                              ║
║ Duration: {features.get('duration_minutes', 0):.2f} minutes ({features.get('duration_seconds', 0):.1f}s)                        ║
║                                                              ║
║ TEMPO & RHYTHM:                                              ║
║   BPM: {features.get('bpm', 0):.1f}                                                    ║
║   Beat Count: {features.get('beat_count', 0):<44} ║
║   Beats/Second: {features.get('beats_per_second', 0):.2f}                                        ║
║                                                              ║
║ KEY & HARMONY:                                               ║
║   Estimated Key: {features.get('estimated_key', 'Unknown'):<41} ║
║   Key Strength: {features.get('key_strength', 0):.3f}                                      ║
║   Harmonic Ratio: {features.get('harmonic_ratio', 0):.3f}                                    ║
║                                                              ║
║ SPECTRAL CHARACTERISTICS:                                    ║
║   Spectral Centroid: {features.get('spectral_centroid_mean', 0):.1f} Hz                          ║
║   Spectral Rolloff: {features.get('spectral_rolloff_mean', 0):.1f} Hz                           ║
║   Spectral Bandwidth: {features.get('spectral_bandwidth_mean', 0):.1f} Hz                        ║
║                                                              ║
║ ENERGY & DYNAMICS:                                           ║
║   RMS Energy: {features.get('rms_mean', 0):.4f}                                       ║
║   Loudness: {features.get('loudness_mean', 0):.1f} dB                                        ║
║   Dynamic Range: {features.get('dynamic_range', 0):.4f}                                   ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary


# Example usage
if __name__ == "__main__":
    extractor = AudioFeatureExtractor()

    # Example: Extract features from an audio file
    # features = extractor.extract_all_features("path/to/audio.mp3")
    # print(extractor.get_feature_summary(features))

    print("AudioFeatureExtractor initialized successfully!")
    print("Available methods:")
    print("  - extract_all_features(file_path)")
    print("  - extract_tempo_features(y, sr)")
    print("  - extract_key_features(y, sr)")
    print("  - extract_spectral_features(y, sr)")
    print("  - extract_mfcc_features(y, sr)")
    print("  - extract_energy_features(y, sr)")
