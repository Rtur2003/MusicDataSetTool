"""
Main Audio Analyzer Pipeline
Combines all features: audio analysis, genre classification, mood detection, and API integrations
"""

import os
import json
from typing import Dict, Optional, List
from datetime import datetime

from features.audio_features import AudioFeatureExtractor
from models.genre_classifier import GenreClassifier
from models.mood_analyzer import MoodAnalyzer
from integrations.spotify_integration import SpotifyIntegration
from integrations.youtube_integration import YouTubeIntegration
from integrations.apple_music_integration import AppleMusicIntegration


class MusicAnalyzer:
    """
    Complete music analysis pipeline combining local audio analysis
    and external API integrations
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the music analyzer

        Args:
            model_dir: Directory containing trained models
        """
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor()
        self.genre_classifier = GenreClassifier()
        self.mood_analyzer = MoodAnalyzer()

        # Load models if available
        if model_dir and os.path.exists(model_dir):
            try:
                self.genre_classifier.load_model(model_dir)
            except:
                print("Genre classifier model not found. Using rule-based prediction.")

            try:
                self.mood_analyzer.load_model(model_dir)
            except:
                print("Mood analyzer model not found. Using rule-based prediction.")

        # Initialize API integrations
        self.spotify = SpotifyIntegration()
        self.youtube = YouTubeIntegration()
        self.apple_music = AppleMusicIntegration()

    def analyze_audio_file(self, file_path: str, include_apis: bool = False) -> Dict:
        """
        Perform complete analysis of an audio file

        Args:
            file_path: Path to audio file
            include_apis: Whether to include API lookups

        Returns:
            Complete analysis dictionary
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {os.path.basename(file_path)}")
        print(f"{'='*60}\n")

        analysis = {
            'file_path': file_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        try:
            # 1. Extract audio features
            print("📊 Extracting audio features...")
            features = self.feature_extractor.extract_all_features(file_path)
            analysis['audio_features'] = features

            # 2. Classify genre
            print("🎸 Classifying genre...")
            genre_result = self.genre_classifier.predict(features)
            analysis['genre'] = genre_result

            # 3. Analyze mood
            print("😊 Analyzing mood...")
            mood_result = self.mood_analyzer.predict(features)
            analysis['mood'] = mood_result

            # 4. API integrations (optional)
            if include_apis:
                analysis['api_data'] = self._fetch_api_data(file_path)

            # 5. Generate summary
            analysis['summary'] = self._generate_summary(analysis)

        except Exception as e:
            analysis['status'] = 'error'
            analysis['error'] = str(e)
            print(f"❌ Error during analysis: {e}")

        return analysis

    def _fetch_api_data(self, file_path: str) -> Dict:
        """
        Fetch data from external APIs

        Args:
            file_path: Audio file path (used to extract track name)

        Returns:
            Dictionary with API data
        """
        api_data = {}

        # Extract track name from filename (basic approach)
        track_name = os.path.splitext(os.path.basename(file_path))[0]

        # Spotify
        if self.spotify.sp:
            print("🎵 Searching Spotify...")
            spotify_results = self.spotify.search_track(track_name, limit=3)
            if spotify_results:
                api_data['spotify'] = {
                    'search_results': spotify_results,
                    'top_match': spotify_results[0] if spotify_results else None
                }

                # Get audio features for top match
                if spotify_results:
                    track_id = spotify_results[0]['id']
                    audio_features = self.spotify.get_audio_features(track_id)
                    if audio_features:
                        api_data['spotify']['audio_features'] = audio_features

        # YouTube
        if self.youtube.youtube:
            print("📺 Searching YouTube...")
            youtube_results = self.youtube.search_music(track_name, max_results=3)
            if youtube_results:
                api_data['youtube'] = {
                    'search_results': youtube_results,
                    'top_match': youtube_results[0] if youtube_results else None
                }

        # Apple Music
        if self.apple_music.token:
            print("🍎 Searching Apple Music...")
            apple_results = self.apple_music.search_songs(track_name, limit=3)
            if apple_results:
                api_data['apple_music'] = {
                    'search_results': apple_results,
                    'top_match': apple_results[0] if apple_results else None
                }

        return api_data

    def _generate_summary(self, analysis: Dict) -> str:
        """
        Generate a human-readable summary

        Args:
            analysis: Complete analysis dictionary

        Returns:
            Formatted summary string
        """
        features = analysis.get('audio_features', {})
        genre = analysis.get('genre', {})
        mood = analysis.get('mood', {})

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║               COMPLETE MUSIC ANALYSIS REPORT                 ║
╠══════════════════════════════════════════════════════════════╣
║ File: {os.path.basename(features.get('file_path', 'Unknown')):<54} ║
║ Analyzed: {analysis['analysis_timestamp'][:19]:<46} ║
║                                                              ║
║ ═══════════════════ AUDIO FEATURES ═══════════════════════   ║
║                                                              ║
║ Duration: {features.get('duration_minutes', 0):.2f} min                                         ║
║ BPM: {features.get('bpm', 0):.1f}                                                    ║
║ Key: {features.get('estimated_key', 'Unknown'):<55} ║
║ Tempo: {features.get('tempo', 'N/A'):<53} ║
║                                                              ║
║ Spectral Centroid: {features.get('spectral_centroid_mean', 0):.1f} Hz                          ║
║ RMS Energy: {features.get('rms_mean', 0):.4f}                                       ║
║ Loudness: {features.get('loudness_mean', 0):.1f} dB                                        ║
║                                                              ║
║ ════════════════════ CLASSIFICATION ════════════════════     ║
║                                                              ║
║ Genre: {genre.get('predicted_genre', 'Unknown').upper():<51} ║
║ Confidence: {genre.get('confidence', 0):.1%}                                         ║
"""

        # Add top 3 genre predictions
        if 'top_3_predictions' in genre:
            summary += "║ Top Predictions:                                             ║\n"
            for i, pred in enumerate(genre['top_3_predictions'][:3], 1):
                summary += f"║   {i}. {pred['genre'].capitalize():<20} ({pred['probability']:.1%})                    ║\n"

        summary += f"""║                                                              ║
║ ═════════════════════ MOOD ANALYSIS ═════════════════════    ║
║                                                              ║
║ Mood: {mood.get('predicted_mood', 'Unknown').upper():<53} ║
║ Confidence: {mood.get('confidence', 0):.1%}                                         ║
"""

        # Valence and Arousal
        if 'valence_arousal' in mood:
            va = mood['valence_arousal']
            summary += f"""║                                                              ║
║ Valence (Positivity): {va['valence']:.2f} ({va['valence_label']})                    ║
║ Arousal (Energy): {va['arousal']:.2f} ({va['arousal_label']})                        ║
║ Emotional Quadrant: {mood.get('emotional_quadrant', 'N/A'):<35} ║
"""

        # Add top mood predictions
        if 'top_3_predictions' in mood:
            summary += "║                                                              ║\n"
            summary += "║ Top Mood Predictions:                                        ║\n"
            for i, pred in enumerate(mood['top_3_predictions'][:3], 1):
                summary += f"║   {i}. {pred['mood'].capitalize():<20} ({pred['probability']:.1%})                    ║\n"

        summary += "╚══════════════════════════════════════════════════════════════╝\n"

        return summary

    def analyze_from_url(self, url: str, platform: str = 'auto') -> Dict:
        """
        Analyze music from a URL (YouTube, Spotify, etc.)

        Args:
            url: URL to the music
            platform: Platform name ('spotify', 'youtube', 'apple_music', 'auto')

        Returns:
            Analysis dictionary
        """
        # Detect platform
        if platform == 'auto':
            if 'spotify.com' in url:
                platform = 'spotify'
            elif 'youtube.com' in url or 'youtu.be' in url:
                platform = 'youtube'
            elif 'apple.com' in url:
                platform = 'apple_music'

        analysis = {
            'url': url,
            'platform': platform,
            'analysis_timestamp': datetime.now().isoformat()
        }

        try:
            if platform == 'spotify':
                # Extract track ID from URL
                track_id = url.split('/')[-1].split('?')[0]
                analysis['track_info'] = self.spotify.get_track_info(track_id)
                analysis['audio_features'] = self.spotify.get_audio_features(track_id)

            elif platform == 'youtube':
                # Extract video ID
                video_id = self.youtube.extract_video_id_from_url(url)
                if video_id:
                    analysis['video_info'] = self.youtube.get_video_info(video_id)

            elif platform == 'apple_music':
                print("Apple Music URL analysis not fully implemented yet.")

        except Exception as e:
            analysis['error'] = str(e)

        return analysis

    def batch_analyze(self, file_paths: List[str],
                     output_file: Optional[str] = None) -> List[Dict]:
        """
        Analyze multiple audio files

        Args:
            file_paths: List of audio file paths
            output_file: Optional JSON file to save results

        Returns:
            List of analysis dictionaries
        """
        results = []

        print(f"\n🎵 Batch Analysis: {len(file_paths)} files\n")

        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] Processing: {os.path.basename(file_path)}")
            try:
                analysis = self.analyze_audio_file(file_path, include_apis=False)
                results.append(analysis)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': str(e)
                })

        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to: {output_file}")

        return results

    def export_analysis(self, analysis: Dict, output_path: str,
                       format: str = 'json'):
        """
        Export analysis to file

        Args:
            analysis: Analysis dictionary
            output_path: Output file path
            format: Export format ('json', 'txt')
        """
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(analysis.get('summary', ''))
                f.write("\n\n" + "="*60 + "\n")
                f.write("FULL ANALYSIS DATA\n")
                f.write("="*60 + "\n\n")
                f.write(json.dumps(analysis, indent=2, ensure_ascii=False))

        print(f"✅ Analysis exported to: {output_path}")

    def compare_tracks(self, file_paths: List[str]) -> Dict:
        """
        Compare multiple tracks

        Args:
            file_paths: List of audio file paths (2-5 tracks)

        Returns:
            Comparison dictionary
        """
        if len(file_paths) < 2:
            raise ValueError("Need at least 2 tracks to compare")

        if len(file_paths) > 5:
            print("Warning: Comparing more than 5 tracks. Limiting to first 5.")
            file_paths = file_paths[:5]

        analyses = []
        for file_path in file_paths:
            analysis = self.analyze_audio_file(file_path, include_apis=False)
            analyses.append(analysis)

        # Compare features
        comparison = {
            'tracks': [os.path.basename(a['file_path']) for a in analyses],
            'genres': [a['genre']['predicted_genre'] for a in analyses],
            'moods': [a['mood']['predicted_mood'] for a in analyses],
            'bpms': [a['audio_features']['bpm'] for a in analyses],
            'keys': [a['audio_features']['estimated_key'] for a in analyses],
            'durations': [a['audio_features']['duration_minutes'] for a in analyses]
        }

        return comparison


# Example usage and CLI entry point
if __name__ == "__main__":
    import sys

    print("🎵 Music Analyzer Tool")
    print("="*60)

    if len(sys.argv) > 1:
        analyzer = MusicAnalyzer()
        file_path = sys.argv[1]

        if os.path.exists(file_path):
            result = analyzer.analyze_audio_file(file_path, include_apis=True)
            print(result['summary'])

            # Optionally save to JSON
            output_file = file_path.replace(os.path.splitext(file_path)[1], '_analysis.json')
            analyzer.export_analysis(result, output_file)
        else:
            print(f"❌ File not found: {file_path}")
    else:
        print("\nUsage:")
        print("  python analyzer.py <audio_file_path>")
        print("\nExample:")
        print("  python analyzer.py music/song.mp3")
