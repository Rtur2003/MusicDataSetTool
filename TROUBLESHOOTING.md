# Troubleshooting Guide

This guide helps you resolve common issues with the Music Dataset Tool.

## Installation Issues

### Problem: pip install fails with "No module named 'numpy'"

**Solution:**
```bash
# Install numpy separately first
pip install numpy
pip install -e .
```

### Problem: librosa installation fails

**Solution:**
Install system dependencies first:
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1 ffmpeg

# macOS
brew install libsndfile ffmpeg

# Windows
# Download and install FFmpeg from https://ffmpeg.org/download.html
```

### Problem: TensorFlow installation issues

**Solution:**
```bash
# For Apple Silicon Macs
pip install tensorflow-macos

# For older systems, use CPU-only version
pip install tensorflow-cpu
```

## Audio Loading Issues

### Problem: "AudioLoadError: Error loading audio file"

**Possible Causes & Solutions:**

1. **Unsupported format**
   ```python
   # Supported formats: .mp3, .wav, .flac, .m4a, .ogg, .aac, .wma
   # Convert your file using ffmpeg
   ffmpeg -i input.xxx output.mp3
   ```

2. **Corrupted file**
   ```bash
   # Check file integrity
   ffmpeg -v error -i file.mp3 -f null -
   ```

3. **File permissions**
   ```bash
   # Check file permissions
   ls -l file.mp3
   chmod 644 file.mp3
   ```

### Problem: "File does not exist" error

**Solution:**
```python
# Use absolute paths
from pathlib import Path
file_path = Path('music/song.mp3').absolute()
analyzer.analyze_audio_file(str(file_path))
```

## API Integration Issues

### Problem: Spotify API not working

**Possible Causes & Solutions:**

1. **Missing credentials**
   ```bash
   # Check .env file
   cat .env
   # Should contain:
   # SPOTIFY_CLIENT_ID=your_id
   # SPOTIFY_CLIENT_SECRET=your_secret
   ```

2. **Invalid credentials**
   - Verify credentials at https://developer.spotify.com/dashboard
   - Regenerate if necessary

3. **Rate limiting**
   - The tool implements automatic retry with backoff
   - Wait a few minutes and try again

### Problem: YouTube API quota exceeded

**Solution:**
```python
# YouTube API has daily quotas
# Reduce the number of requests or wait until the next day
# You can also create additional API keys for higher quotas
```

### Problem: Apple Music token generation fails

**Possible Causes & Solutions:**

1. **Missing private key file**
   ```bash
   # Check if file exists
   ls -l path/to/key.p8
   ```

2. **Incorrect key format**
   - Ensure it's a valid .p8 file from Apple Developer
   - File should start with "-----BEGIN PRIVATE KEY-----"

3. **Expired developer account**
   - Check your Apple Developer Program membership

## Model Issues

### Problem: "Model not found" warning

**Solution:**
This is expected if you haven't trained models yet. The tool will use rule-based prediction:
```python
# To train models, you need labeled data
# X = feature arrays, y = labels
classifier.train(X, y, epochs=100)
classifier.save_model('models/')
```

### Problem: Low prediction accuracy

**Solutions:**

1. **Train with more data**
   ```python
   # Use larger dataset
   # Ensure balanced classes
   ```

2. **Adjust hyperparameters**
   ```python
   classifier.train(X, y, epochs=200, batch_size=64)
   ```

3. **Feature engineering**
   - Ensure audio quality is good
   - Check for consistent sample rates

## Memory Issues

### Problem: Out of memory during batch processing

**Solutions:**

1. **Process fewer files at once**
   ```python
   # Process in smaller batches
   batch_size = 10
   for i in range(0, len(files), batch_size):
       batch = files[i:i+batch_size]
       analyzer.batch_analyze(batch)
   ```

2. **Reduce sample rate**
   ```python
   from src.features.audio_features import AudioFeatureExtractor
   extractor = AudioFeatureExtractor(sample_rate=16000)  # Lower than default
   ```

## Performance Issues

### Problem: Analysis is too slow

**Solutions:**

1. **Disable API lookups**
   ```python
   result = analyzer.analyze_audio_file('song.mp3', include_apis=False)
   ```

2. **Use caching**
   ```python
   from src.utils.cache import cached
   
   @cached(ttl_seconds=3600)
   def analyze_with_cache(file_path):
       return analyzer.analyze_audio_file(file_path)
   ```

3. **Process on GPU**
   ```python
   # Ensure TensorFlow can use GPU
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

## Testing Issues

### Problem: Tests fail with import errors

**Solution:**
```bash
# Install in development mode
pip install -e ".[dev]"

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Problem: Tests fail due to missing audio files

**Solution:**
Tests use temporary files. If you see this error:
```bash
# Generate test audio files
python -c "
import numpy as np
import soundfile as sf
sr = 22050
duration = 3
audio = np.random.randn(sr * duration)
sf.write('test_audio.wav', audio, sr)
"
```

## Logging and Debugging

### Enable debug logging

```python
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG, log_file='debug.log')
```

### Check log files

```bash
# View recent logs
tail -f debug.log

# Search for errors
grep ERROR debug.log
```

## Common Error Messages

### "ValidationError: File path cannot be empty"
**Cause:** No file path provided  
**Solution:** Provide a valid file path

### "FeatureExtractionError: Error extracting tempo features"
**Cause:** Audio file is too short or silent  
**Solution:** Use audio files at least 3 seconds long with actual content

### "ModelError: Model must be trained before saving"
**Cause:** Attempting to save untrained model  
**Solution:** Train the model first using `classifier.train(X, y)`

### "APIError: Spotify search failed"
**Cause:** Network issue or invalid credentials  
**Solution:** Check internet connection and API credentials

## Environment-Specific Issues

### macOS Issues

**Problem:** "ImportError: cannot import name 'sndfile'"
```bash
# Install via homebrew
brew install libsndfile
pip install soundfile --force-reinstall
```

### Windows Issues

**Problem:** Long path names
```bash
# Enable long path support in Windows
# Run as administrator:
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
```

### Linux Issues

**Problem:** Permission denied errors
```bash
# Fix permissions
chmod +x src/cli.py
sudo chown -R $USER:$USER .
```

## Getting Help

If you're still experiencing issues:

1. **Check existing issues:** https://github.com/Rtur2003/MusicDataSetTool/issues
2. **Search discussions:** Look for similar problems
3. **Create new issue:** Provide:
   - Python version (`python --version`)
   - OS and version
   - Full error message and traceback
   - Minimal code to reproduce
   - Log files (if applicable)

## Useful Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "librosa|tensorflow|spotipy"

# Test audio file
python -c "import librosa; y, sr = librosa.load('file.mp3'); print(f'Loaded {len(y)} samples at {sr} Hz')"

# Verify installation
python -c "from src.analyzer import MusicAnalyzer; print('Installation OK')"

# Clear cache
python -c "from src.utils.cache import SimpleCache; SimpleCache().clear()"

# Run specific test
pytest tests/test_validators.py -v

# Check code quality
flake8 src/
black --check src/
```
