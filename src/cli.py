"""
Command-line interface for Music Dataset Tool.
Provides a user-friendly CLI for analyzing audio files.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from .analyzer import MusicAnalyzer
from .config import get_config
from .utils.logger import get_logger
from .utils.exceptions import MusicAnalyzerError

logger = get_logger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog='music-analyze',
        description='Comprehensive music analysis tool with ML, audio processing, and API integrations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  music-analyze song.mp3
  
  # Analyze with API lookups
  music-analyze song.mp3 --include-apis
  
  # Batch analyze multiple files
  music-analyze song1.mp3 song2.mp3 song3.mp3 --output results.json
  
  # Compare tracks
  music-analyze --compare track1.mp3 track2.mp3 track3.mp3
  
  # Export to text format
  music-analyze song.mp3 --format txt --output analysis.txt
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        type=str,
        help='Audio file(s) to analyze'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path for results (default: stdout)'
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'txt'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--include-apis',
        action='store_true',
        help='Include API lookups (Spotify, YouTube, Apple Music)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple tracks instead of analyzing individually'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: stdout only)'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser


def analyze_single(analyzer: MusicAnalyzer, file_path: str, 
                   include_apis: bool, output: Optional[str], 
                   format: str) -> int:
    """
    Analyze a single audio file.
    
    Args:
        analyzer: MusicAnalyzer instance
        file_path: Path to audio file
        include_apis: Whether to include API lookups
        output: Output file path (None for stdout)
        format: Output format ('json' or 'txt')
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        result = analyzer.analyze_audio_file(file_path, include_apis=include_apis)
        
        if result['status'] == 'error':
            logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            return 1
        
        if output:
            analyzer.export_analysis(result, output, format=format)
            logger.info(f"Results saved to {output}")
        else:
            if format == 'json':
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(result.get('summary', ''))
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to analyze {file_path}: {str(e)}")
        return 1


def analyze_batch(analyzer: MusicAnalyzer, file_paths: list, 
                  include_apis: bool, output: Optional[str]) -> int:
    """
    Analyze multiple audio files.
    
    Args:
        analyzer: MusicAnalyzer instance
        file_paths: List of audio file paths
        include_apis: Whether to include API lookups
        output: Output file path
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        results = analyzer.batch_analyze(file_paths, output_file=output)
        
        if not output:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        failed = sum(1 for r in results if r.get('status') == 'error')
        if failed > 0:
            logger.warning(f"{failed} out of {len(results)} files failed to analyze")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        return 1


def compare_tracks(analyzer: MusicAnalyzer, file_paths: list, 
                  output: Optional[str]) -> int:
    """
    Compare multiple audio tracks.
    
    Args:
        analyzer: MusicAnalyzer instance
        file_paths: List of audio file paths
        output: Output file path
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        comparison = analyzer.compare_tracks(file_paths)
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            logger.info(f"Comparison saved to {output}")
        else:
            print(json.dumps(comparison, indent=2, ensure_ascii=False))
        
        return 0
        
    except Exception as e:
        logger.error(f"Track comparison failed: {str(e)}")
        return 1


def main(argv=None):
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv)
        
    Returns:
        Exit code
    """
    parser = setup_parser()
    args = parser.parse_args(argv)
    
    # Configure logging
    import logging
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    if args.log_file:
        from .utils.logger import get_logger
        logger = get_logger(__name__, level=log_level, log_file=args.log_file)
    
    try:
        # Initialize analyzer
        logger.info("Initializing MusicAnalyzer...")
        analyzer = MusicAnalyzer(model_dir=args.model_dir)
        
        # Determine operation mode
        if args.compare:
            if len(args.files) < 2:
                logger.error("Comparison requires at least 2 files")
                return 1
            return compare_tracks(analyzer, args.files, args.output)
        
        elif len(args.files) == 1:
            return analyze_single(
                analyzer, args.files[0], 
                args.include_apis, args.output, args.format
            )
        
        else:
            return analyze_batch(
                analyzer, args.files,
                args.include_apis, args.output
            )
            
    except MusicAnalyzerError as e:
        logger.error(f"Analysis error: {str(e)}")
        return 1
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
