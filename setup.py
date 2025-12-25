"""Setup configuration for Music Dataset Tool."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='music-dataset-tool',
    version='1.0.0',
    author='Hasan Arthur Altuntaş',
    author_email='',
    description='A comprehensive music analysis tool combining ML, audio signal processing, and streaming platform integrations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Rtur2003/MusicDataSetTool',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'librosa>=0.10.0',
        'soundfile>=0.12.1',
        'pydub>=0.25.1',
        'audioread>=3.0.0',
        'tensorflow>=2.13.0',
        'scikit-learn>=1.3.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'spotipy>=2.23.0',
        'google-api-python-client>=2.95.0',
        'google-auth-httplib2>=0.1.0',
        'google-auth-oauthlib>=1.0.0',
        'requests>=2.31.0',
        'python-dotenv>=1.0.0',
        'tqdm>=4.65.0',
        'joblib>=1.3.0',
        'PyJWT>=2.8.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'viz': [
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'plotly>=5.15.0',
        ],
        'advanced': [
            'essentia>=2.1b6.dev1034',
            'madmom>=0.16.1',
            'music21>=9.1.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'music-analyze=src.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.yml', '*.yaml'],
    },
)
