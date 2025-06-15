# Spectrogram Visualization System

A sophisticated web application for analyzing non-stationary signals using Short-Time Fourier Transform (STFT). This tool provides comprehensive visualization and analysis capabilities for speech, music, and EEG signals with a modern dark-themed interface.

## Features

### 🎤 Speech Analysis

- Upload and analyze speech recordings (WAV, MP3)
- Visualize phonetic patterns and voice characteristics
- Interactive waveform and spectrogram displays

### 🎵 Music Analysis

- Process musical recordings and compositions
- Analyze frequency patterns and musical notes
- Customizable color maps for spectrograms

### 🧠 EEG Analysis

- Import EEG data from CSV files
- Examine brain wave patterns and rhythms
- Real-time signal processing and visualization

## Technical Specifications

### Core Technologies

- **Frontend**: Streamlit with custom CSS styling
- **Signal Processing**: SciPy, LibROSA
- **Visualization**: Matplotlib, Plotly
- **Data Handling**: NumPy, Pandas

### Signal Processing Features

- Short-Time Fourier Transform (STFT) implementation
- Adaptive window sizing for optimal frequency resolution
- Multiple colormap options for spectrogram visualization
- Real-time signal statistics calculation

### Visualization Capabilities

- **Static Plots**: High-quality matplotlib waveforms and spectrograms
- **Interactive Plots**: Plotly-based interactive spectrograms with zoom and hover
- **Real-time Updates**: Dynamic visualization updates as parameters change
- **Dark Theme**: Modern, eye-friendly dark interface

## Installation

### Prerequisites

```bash
pip install streamlit numpy pandas scipy matplotlib librosa plotly streamlit-aggrid
```

### Quick Start

1. Clone the repository
2. Install dependencies
3. Run the application:

```bash
streamlit run app.py
```

## Usage

### File Upload

- **Audio Files**: Supported formats - WAV, MP3
- **EEG Data**: CSV files with numeric columns
- **Processing**: Automatic format detection and conversion

### Analysis Features

- **Waveform Display**: Time-domain signal visualization
- **Spectrogram Generation**: Frequency-time analysis
- **Interactive Exploration**: Zoom, pan, and hover for detailed inspection
- **Signal Statistics**: Duration, sampling rate, amplitude metrics

### Customization Options

- **Color Maps**: viridis, plasma, inferno, magma, cividis
- **Window Parameters**: Automatic optimization based on signal length
- **Display Options**: Responsive layout with customizable plots

## Signal Processing Details

### STFT Implementation

- **Window Function**: Hamming window (default)
- **Overlap**: 50% overlap between windows
- **FFT Size**: Adaptive based on signal length (minimum 1024 points)
- **Frequency Resolution**: Optimized for signal characteristics

### Supported Signal Types

- **Speech**: Sampling rates up to 48 kHz
- **Music**: Full audio spectrum analysis
- **EEG**: Typical brain wave frequencies (0.5-100 Hz)

## File Structure

```
spectrogram-analyzer/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Configuration

### Session State Management

- Processed data persistence across interactions
- Multiple signal comparison capabilities
- Efficient memory usage with data caching

### Performance Optimization

- Matplotlib backend configuration for server environments
- Efficient STFT computation with optimized parameters
- Responsive UI with loading indicators

## Dependencies

### Core Libraries

```python
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
librosa>=0.10.0
plotly>=5.15.0
streamlit-aggrid>=0.3.4
```

### System Requirements

- Python 3.8+
- Sufficient RAM for audio processing (2GB+ recommended)
- Modern web browser with JavaScript enabled

## Error Handling

### Robust Processing

- Automatic format detection and conversion
- Graceful error handling for corrupted files
- User-friendly error messages and suggestions

### Validation

- File format verification
- Data integrity checks
- Memory usage monitoring

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper testing
4. Submit pull request with detailed description

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Include docstrings for functions
- Maintain consistent formatting

## License

© 2023 Signal Processing Lab • All rights reserved

## Support

For issues, questions, or feature requests, please create an issue in the project repository.

---

**Note**: This application is designed for educational and research purposes. For production use, consider additional security measures and scalability optimizations.
