import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft, welch, hilbert, butter, filtfilt
from scipy.stats import kurtosis, skew, entropy
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import librosa
import librosa.display
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_aggrid import AgGrid
import time
import io

# Configure page with dark theme
st.set_page_config(
    page_title="Advanced Spectrogram Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced dark theme CSS styling
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e6ed;
    }
    
    /* Custom containers */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.2rem !important;
        margin-top: 0.5rem !important;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #232c4b 0%, #2d3561 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border: 1px solid rgba(103, 126, 234, 0.2);
        transition: all 0.3s ease;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(103, 126, 234, 0.4);
        border-color: rgba(103, 126, 234, 0.5);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: #667eea !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .feature-desc {
        color: rgba(224, 230, 237, 0.8) !important;
        font-size: 0.9rem !important;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin: 2rem 0 1rem 0 !important;
        text-align: center;
    }
    
    /* Analysis containers */
    .analysis-container {
        background: rgba(35, 44, 75, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(103, 126, 234, 0.2);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Comparison panel */
    .comparison-panel {
        background: linear-gradient(135deg, #232c4b 0%, #2d3561 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid rgba(103, 126, 234, 0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Upload areas */
    .stFileUploader > div {
        background: rgba(35, 44, 75, 0.6) !important;
        border: 2px dashed rgba(103, 126, 234, 0.5) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(103, 126, 234, 0.8) !important;
        background: rgba(35, 44, 75, 0.8) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(103, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(103, 126, 234, 0.5) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(35, 44, 75, 0.6);
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(224, 230, 237, 0.7);
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(35, 44, 75, 0.8) !important;
        border: 1px solid rgba(103, 126, 234, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #232c4b 0%, #2d3561 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(103, 126, 234, 0.2);
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(103, 126, 234, 0.3);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(46, 204, 113, 0.2) !important;
        border: 1px solid rgba(46, 204, 113, 0.5) !important;
        border-radius: 10px !important;
    }
    
    .stError {
        background: rgba(231, 76, 60, 0.2) !important;
        border: 1px solid rgba(231, 76, 60, 0.5) !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #232c4b 0%, #2d3561 100%) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Spinner */
    .stSpinner {
        color: #667eea !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(35, 44, 75, 0.8) !important;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {
        'speech': None,
        'music': None,
        'eeg': None
    }

if 'comparison_signals' not in st.session_state:
    st.session_state.comparison_signals = []

# Enhanced helper functions
def process_audio(file, signal_type):
    """Process audio files using librosa for better handling"""
    try:
        y, sr = librosa.load(file, sr=None)
        return y, sr
    except Exception as e:
        st.error(f"Error processing {signal_type} file: {str(e)}")
        return None, None

def process_eeg(file):
    """Process EEG data from CSV file"""
    try:
        df = pd.read_csv(file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in EEG data")
        data = df[numeric_cols[0]].values
        return data, 250  # 250 Hz default for EEG
    except Exception as e:
        st.error(f"Error processing EEG file: {str(e)}")
        st.warning("Please ensure your EEG file is a CSV with at least one numeric column")
        return None, None

def extract_time_domain_features(data, sr):
    """Extract comprehensive time domain features"""
    features = {}
    
    # Basic statistics
    features['Mean'] = np.mean(data)
    features['Standard Deviation'] = np.std(data)
    features['Variance'] = np.var(data)
    features['RMS'] = np.sqrt(np.mean(data**2))
    features['Peak-to-Peak'] = np.ptp(data)
    features['Skewness'] = skew(data)
    features['Kurtosis'] = kurtosis(data)
    
    # Zero crossings
    zero_crossings = np.where(np.diff(np.signbit(data)))[0]
    features['Zero Crossing Rate'] = len(zero_crossings) / len(data)
    
    # Energy and power
    features['Energy'] = np.sum(data**2)
    features['Power'] = features['Energy'] / len(data)
    
    # Envelope features
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)
    features['Envelope Mean'] = np.mean(envelope)
    features['Envelope Std'] = np.std(envelope)
    
    # Temporal features
    features['Duration (s)'] = len(data) / sr
    features['Peak Location (s)'] = np.argmax(np.abs(data)) / sr
    
    return features

def extract_frequency_domain_features(data, sr):
    """Extract comprehensive frequency domain features"""
    features = {}
    
    # FFT
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1/sr)
    magnitude = np.abs(fft)
    
    # Keep only positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = magnitude[pos_mask]
    
    # Spectral features
    features['Spectral Centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
    features['Spectral Spread'] = np.sqrt(np.sum(((freqs - features['Spectral Centroid'])**2) * magnitude) / np.sum(magnitude))
    features['Spectral Rolloff'] = freqs[np.where(np.cumsum(magnitude) >= 0.85 * np.sum(magnitude))[0][0]]
    
    # Spectral flux
    features['Spectral Flux'] = np.sum(np.diff(magnitude)**2)
    
    # Dominant frequency
    features['Dominant Frequency'] = freqs[np.argmax(magnitude)]
    features['Peak Magnitude'] = np.max(magnitude)
    
    # Frequency bands analysis
    nyquist = sr / 2
    low_freq = magnitude[(freqs >= 0) & (freqs <= nyquist * 0.25)]
    mid_freq = magnitude[(freqs > nyquist * 0.25) & (freqs <= nyquist * 0.75)]
    high_freq = magnitude[(freqs > nyquist * 0.75) & (freqs <= nyquist)]
    
    total_energy = np.sum(magnitude**2)
    features['Low Freq Energy (%)'] = (np.sum(low_freq**2) / total_energy) * 100 if total_energy > 0 else 0
    features['Mid Freq Energy (%)'] = (np.sum(mid_freq**2) / total_energy) * 100 if total_energy > 0 else 0
    features['High Freq Energy (%)'] = (np.sum(high_freq**2) / total_energy) * 100 if total_energy > 0 else 0
    
    return features

def plot_waveform(data, sr, title):
    """Plot waveform with dark theme styling"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#232c4b')
    time = np.arange(0, len(data)) / sr
    ax.plot(time, data, color='#667eea', linewidth=1.5, alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#e0e6ed')
    ax.set_xlabel('Time (s)', fontsize=12, color='#e0e6ed')
    ax.set_ylabel('Amplitude', fontsize=12, color='#e0e6ed')
    ax.grid(True, linestyle='--', alpha=0.3, color='#667eea')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#e0e6ed')
    for spine in ax.spines.values():
        spine.set_edgecolor('#667eea')
    fig.tight_layout()
    return fig

def plot_fft(data, sr, title):
    """Plot FFT with dark theme"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#232c4b')
    
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1/sr)
    magnitude = np.abs(fft)
    
    # Keep only positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = magnitude[pos_mask]
    
    ax.plot(freqs, 20 * np.log10(magnitude + 1e-10), color='#764ba2', linewidth=1.5)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#e0e6ed')
    ax.set_xlabel('Frequency (Hz)', fontsize=12, color='#e0e6ed')
    ax.set_ylabel('Magnitude (dB)', fontsize=12, color='#e0e6ed')
    ax.grid(True, linestyle='--', alpha=0.3, color='#667eea')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#e0e6ed')
    for spine in ax.spines.values():
        spine.set_edgecolor('#667eea')
    fig.tight_layout()
    return fig

def plot_psd(data, sr, title):
    """Plot Power Spectral Density with dark theme"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#232c4b')
    
    freqs, psd = welch(data, sr, nperseg=min(1024, len(data)//4))
    ax.semilogy(freqs, psd, color='#667eea', linewidth=1.5)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#e0e6ed')
    ax.set_xlabel('Frequency (Hz)', fontsize=12, color='#e0e6ed')
    ax.set_ylabel('Power Spectral Density', fontsize=12, color='#e0e6ed')
    ax.grid(True, linestyle='--', alpha=0.3, color='#667eea')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#e0e6ed')
    for spine in ax.spines.values():
        spine.set_edgecolor('#667eea')
    fig.tight_layout()
    return fig

def plot_spectrogram(data, sr, title, cmap='viridis'):
    """Plot spectrogram using STFT with dark theme"""
    plt.style.use('dark_background')
    nperseg = min(1024, len(data) // 10)
    f, t, Zxx = stft(data, fs=sr, nperseg=nperseg)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#232c4b')
    pcm = ax.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx) + 1e-10), 
                       cmap=cmap, shading='gouraud')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#e0e6ed')
    ax.set_xlabel('Time (s)', fontsize=12, color='#e0e6ed')
    ax.set_ylabel('Frequency (Hz)', fontsize=12, color='#e0e6ed')
    cbar = fig.colorbar(pcm, ax=ax, label='Intensity (dB)')
    cbar.ax.yaxis.set_tick_params(color='#e0e6ed')
    cbar.outline.set_edgecolor('#667eea')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e6ed')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#e0e6ed')
    for spine in ax.spines.values():
        spine.set_edgecolor('#667eea')
    fig.tight_layout()
    return fig

def plot_interactive_spectrogram(data, sr):
    """Interactive spectrogram using Plotly with dark theme"""
    nperseg = min(1024, len(data) // 10)
    f, t, Zxx = stft(data, fs=sr, nperseg=nperseg)
    Z = 20 * np.log10(np.abs(Zxx) + 1e-10)
    
    fig = go.Figure(data=go.Heatmap(
        x=t,
        y=f,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Intensity (dB)')
    ))
    
    fig.update_layout(
        title='Interactive Spectrogram',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=500,
        template='plotly_dark',
        margin=dict(l=50, r=50, b=50, t=80),
        hovermode='closest',
        plot_bgcolor='rgba(35, 44, 75, 0.8)',
        paper_bgcolor='rgba(35, 44, 75, 0.8)',
        font=dict(color='#e0e6ed')
    )
    return fig

def plot_combined_analysis(data, sr, signal_name):
    """Create combined time and frequency domain visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Time Domain', 'Frequency Domain (FFT)', 'Power Spectral Density', 'Spectrogram'],
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # Time domain
    time = np.arange(0, len(data)) / sr
    fig.add_trace(go.Scatter(x=time, y=data, name='Signal', line=dict(color='#667eea')), row=1, col=1)
    
    # FFT
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1/sr)
    magnitude = np.abs(fft)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = magnitude[pos_mask]
    fig.add_trace(go.Scatter(x=freqs, y=20*np.log10(magnitude + 1e-10), name='FFT', line=dict(color='#764ba2')), row=1, col=2)
    
    # PSD
    freqs_psd, psd = welch(data, sr, nperseg=min(1024, len(data)//4))
    fig.add_trace(go.Scatter(x=freqs_psd, y=psd, name='PSD', line=dict(color='#667eea')), row=2, col=1)
    
    # Spectrogram
    nperseg = min(1024, len(data) // 10)
    f, t, Zxx = stft(data, fs=sr, nperseg=nperseg)
    Z = 20 * np.log10(np.abs(Zxx) + 1e-10)
    fig.add_trace(go.Heatmap(x=t, y=f, z=Z, colorscale='Viridis', showscale=False), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=f'Combined Analysis - {signal_name}',
        height=800,
        template='plotly_dark',
        plot_bgcolor='rgba(35, 44, 75, 0.8)',
        paper_bgcolor='rgba(35, 44, 75, 0.8)',
        font=dict(color='#e0e6ed'),
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="PSD", row=2, col=1, type="log")
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=2)
    
    return fig

def display_feature_comparison(features_dict):
    """Display features in an organized table format"""
    df = pd.DataFrame.from_dict(features_dict, orient='index', columns=['Value'])
    df.reset_index(inplace=True)
    df.columns = ['Feature', 'Value']
    
    # Format numerical values
    df['Value'] = df['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else str(x))
    
    return df

def show_analysis(signal_type, data, sr):
    """Display comprehensive analysis for a signal type"""
    with st.container():
        st.markdown(f'<div class="analysis-container">', unsafe_allow_html=True)
        
        st.subheader(f"{signal_type.capitalize()} Analysis", anchor=False)
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "⏱️ Time Domain", "📊 Frequency Domain", "🔍 Combined View", "📋 Features"])
        
        with tab1:
            # Overview with basic visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Waveform")
                with st.spinner('Generating waveform...'):
                    fig = plot_waveform(data, sr, f"{signal_type.capitalize()} Signal")
                    st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Spectrogram")
                cmap = st.selectbox(
                    "Color map", 
                    ["viridis", "plasma", "inferno", "magma", "cividis"],
                    key=f"{signal_type}_cmap"
                )
                with st.spinner('Generating spectrogram...'):
                    fig = plot_spectrogram(data, sr, f"{signal_type.capitalize()} Spectrogram", cmap)
                    st.pyplot(fig, use_container_width=True)
            
            # Interactive spectrogram
            st.markdown("### Interactive Visualization")
            with st.spinner('Creating interactive plot...'):
                fig = plot_interactive_spectrogram(data, sr)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Time domain analysis
            st.markdown("### Time Domain Analysis")
            
            # Extract time domain features
            time_features = extract_time_domain_features(data, sr)
            
            # Display waveform with enhanced features
            fig = plot_waveform(data, sr, f"{signal_type.capitalize()} - Time Domain")
            st.pyplot(fig, use_container_width=True)
            
            # Feature display
            st.markdown("#### Time Domain Features")
            feature_df = display_feature_comparison(time_features)
            st.dataframe(feature_df, use_container_width=True)
            
            # Statistical summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">RMS Value</h4>
                    <p style="font-size: 1.2rem; margin: 0;">{time_features['RMS']:.6f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">Zero Crossing Rate</h4>
                    <p style="font-size: 1.2rem; margin: 0;">{time_features['Zero Crossing Rate']:.6f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">Kurtosis</h4>
                    <p style="font-size: 1.2rem; margin: 0;">{time_features['Kurtosis']:.6f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            # Frequency domain analysis
            st.markdown("### Frequency Domain Analysis")
            
            # Extract frequency domain features
            freq_features = extract_frequency_domain_features(data, sr)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### FFT Analysis")
                fig = plot_fft(data, sr, f"{signal_type.capitalize()} - FFT")
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Power Spectral Density")
                fig = plot_psd(data, sr, f"{signal_type.capitalize()} - PSD")
                st.pyplot(fig, use_container_width=True)
            
            # Feature display
            st.markdown("#### Frequency Domain Features")
            freq_df = display_feature_comparison(freq_features)
            st.dataframe(freq_df, use_container_width=True)
            
            # Key frequency metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">Spectral Centroid</h4>
                    <p style="font-size: 1.2rem; margin: 0;">{freq_features['Spectral Centroid']:.2f} Hz</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">Dominant Frequency</h4>
                    <p style="font-size: 1.2rem; margin: 0;">{freq_features['Dominant Frequency']:.2f} Hz</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">Spectral Rolloff</h4>
                    <p style="font-size: 1.2rem; margin: 0;">{freq_features['Spectral Rolloff']:.2f} Hz</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            # Combined view
            st.markdown("### Combined Analysis View")
            with st.spinner('Creating combined visualization...'):
                fig = plot_combined_analysis(data, sr, signal_type.capitalize())
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            # Features summary
            st.markdown("### Complete Feature Summary")
            
            # Combine all features
            time_features = extract_time_domain_features(data, sr)
            freq_features = extract_frequency_domain_features(data, sr)
            
            all_features = {**time_features, **freq_features}
            
            # Create comprehensive feature table
            feature_df = display_feature_comparison(all_features)
            
            # Add feature categories
            time_domain_features = list(time_features.keys())
            freq_domain_features = list(freq_features.keys())
            
            feature_df['Category'] = feature_df['Feature'].apply(
                lambda x: 'Time Domain' if x in time_domain_features else 'Frequency Domain'
            )
            
            # Reorder columns
            feature_df = feature_df[['Category', 'Feature', 'Value']]
            
            st.dataframe(feature_df, use_container_width=True)
            
            # Download feature data
            csv = feature_df.to_csv(index=False)
            st.download_button(
                label="Download Features as CSV",
                data=csv,
                file_name=f"{signal_type}_features.csv",
                mime="text/csv"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

def multi_signal_comparison():
    """Multi-signal comparison panel"""
    st.markdown('<h2 class="section-header">🔍 Multi-Signal Comparison Panel</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="comparison-panel">', unsafe_allow_html=True)
        
        st.markdown("### Compare Multiple Signals Simultaneously")
        st.info("Upload up to 5 signals for comparative analysis")
        
        # Signal upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            comparison_files = st.file_uploader(
                "Upload signals for comparison (WAV, MP3, CSV)",
                type=["wav", "mp3", "csv"],
                accept_multiple_files=True,
                key="comparison_uploader"
            )
        
        with col2:
            if st.button("Clear Comparison Signals", type="secondary"):
                st.session_state.comparison_signals = []
                st.rerun()
        
        if comparison_files and len(comparison_files) <= 5:
            signals_data = []
            
            for i, file in enumerate(comparison_files):
                with st.spinner(f'Processing signal {i+1}...'):
                    if file.name.endswith('.csv'):
                        data, sr = process_eeg(file)
                        signal_type = f"Signal_{i+1}_EEG"
                    else:
                        data, sr = process_audio(file, f"signal_{i+1}")
                        signal_type = f"Signal_{i+1}_Audio"
                    
                    if data is not None:
                        signals_data.append({
                            'name': file.name,
                            'type': signal_type,
                            'data': data,
                            'sr': sr
                        })
            
            if signals_data:
                # Comparison visualization
                st.markdown("### Comparison Visualization")
                
                # Create comparison tabs
                comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs(["📊 Waveforms", "🎵 Spectrograms", "📈 Features", "📋 Statistics"])
                
                with comp_tab1:
                    # Plot all waveforms together
                    fig = make_subplots(
                        rows=len(signals_data), cols=1,
                        subplot_titles=[f"{signal['name']}" for signal in signals_data],
                        vertical_spacing=0.08
                    )
                    
                    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
                    
                    for i, signal in enumerate(signals_data):
                        time = np.arange(0, len(signal['data'])) / signal['sr']
                        fig.add_trace(
                            go.Scatter(
                                x=time, 
                                y=signal['data'], 
                                name=signal['name'],
                                line=dict(color=colors[i % len(colors)])
                            ), 
                            row=i+1, col=1
                        )
                        fig.update_xaxes(title_text="Time (s)", row=i+1, col=1)
                        fig.update_yaxes(title_text="Amplitude", row=i+1, col=1)
                    
                    fig.update_layout(
                        height=300 * len(signals_data),
                        title="Multi-Signal Waveform Comparison",
                        template='plotly_dark',
                        plot_bgcolor='rgba(35, 44, 75, 0.8)',
                        paper_bgcolor='rgba(35, 44, 75, 0.8)',
                        font=dict(color='#e0e6ed'),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with comp_tab2:
                    # Spectrograms comparison
                    st.markdown("### Spectrogram Comparison")
                    
                    cols = st.columns(min(len(signals_data), 2))
                    
                    for i, signal in enumerate(signals_data):
                        with cols[i % 2]:
                            st.markdown(f"#### {signal['name']}")
                            fig = plot_spectrogram(
                                signal['data'], 
                                signal['sr'], 
                                f"Spectrogram - {signal['name']}"
                            )
                            st.pyplot(fig, use_container_width=True)
                
                with comp_tab3:
                    # Features comparison
                    st.markdown("### Feature Comparison")
                    
                    # Extract features for all signals
                    comparison_features = {}
                    
                    for signal in signals_data:
                        time_features = extract_time_domain_features(signal['data'], signal['sr'])
                        freq_features = extract_frequency_domain_features(signal['data'], signal['sr'])
                        all_features = {**time_features, **freq_features}
                        comparison_features[signal['name']] = all_features
                    
                    # Create comparison DataFrame
                    comp_df = pd.DataFrame(comparison_features).T
                    comp_df = comp_df.round(6)
                    
                    st.dataframe(comp_df, use_container_width=True)
                    
                    # Feature visualization
                    st.markdown("#### Key Feature Comparison")
                    
                    key_features = ['RMS', 'Spectral Centroid', 'Dominant Frequency', 'Zero Crossing Rate']
                    available_features = [f for f in key_features if f in comp_df.columns]
                    
                    if available_features:
                        selected_feature = st.selectbox("Select feature to compare:", available_features)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(comparison_features.keys()),
                            y=[comparison_features[signal][selected_feature] for signal in comparison_features],
                            marker_color='#667eea'
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_feature} Comparison",
                            xaxis_title="Signals",
                            yaxis_title=selected_feature,
                            template='plotly_dark',
                            plot_bgcolor='rgba(35, 44, 75, 0.8)',
                            paper_bgcolor='rgba(35, 44, 75, 0.8)',
                            font=dict(color='#e0e6ed')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download comparison data
                    csv = comp_df.to_csv()
                    st.download_button(
                        label="Download Comparison Data as CSV",
                        data=csv,
                        file_name="signal_comparison.csv",
                        mime="text/csv"
                    )
                
                with comp_tab4:
                    # Statistical summary
                    st.markdown("### Statistical Summary")
                    
                    # Create summary statistics
                    summary_stats = {}
                    
                    for signal in signals_data:
                        stats = {
                            'Duration (s)': len(signal['data']) / signal['sr'],
                            'Sampling Rate (Hz)': signal['sr'],
                            'Mean': np.mean(signal['data']),
                            'Std Dev': np.std(signal['data']),
                            'Min': np.min(signal['data']),
                            'Max': np.max(signal['data']),
                            'RMS': np.sqrt(np.mean(signal['data']**2))
                        }
                        summary_stats[signal['name']] = stats
                    
                    summary_df = pd.DataFrame(summary_stats).T
                    summary_df = summary_df.round(6)
                    
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Visual summary
                    st.markdown("#### Signal Characteristics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Duration comparison
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(summary_stats.keys()),
                            y=[summary_stats[signal]['Duration (s)'] for signal in summary_stats],
                            marker_color='#764ba2'
                        ))
                        
                        fig.update_layout(
                            title="Signal Duration Comparison",
                            xaxis_title="Signals",
                            yaxis_title="Duration (s)",
                            template='plotly_dark',
                            plot_bgcolor='rgba(35, 44, 75, 0.8)',
                            paper_bgcolor='rgba(35, 44, 75, 0.8)',
                            font=dict(color='#e0e6ed')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # RMS comparison
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(summary_stats.keys()),
                            y=[summary_stats[signal]['RMS'] for signal in summary_stats],
                            marker_color='#f093fb'
                        ))
                        
                        fig.update_layout(
                            title="RMS Value Comparison",
                            xaxis_title="Signals",
                            yaxis_title="RMS",
                            template='plotly_dark',
                            plot_bgcolor='rgba(35, 44, 75, 0.8)',
                            paper_bgcolor='rgba(35, 44, 75, 0.8)',
                            font=dict(color='#e0e6ed')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        elif comparison_files and len(comparison_files) > 5:
            st.warning("Please upload a maximum of 5 signals for comparison.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Main app
def main():
    # Header with gradient
    st.markdown("""
    <div class="main-header">
        <h1>Advanced Spectrogram Analyzer</h1>
        <p>Comprehensive Time & Frequency Domain Analysis with Multi-Signal Comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced feature cards
    cols = st.columns(4)
    features = [
        ("🎤", "Speech Analysis", "Voice patterns & phonetics"),
        ("🎵", "Music Analysis", "Musical notes & frequencies"),
        ("🧠", "EEG Analysis", "Brain wave patterns"),
        ("🔍", "Multi-Comparison", "Compare up to 5 signals")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <h3 class="feature-title">{title}</h3>
                    <p class="feature-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Multi-signal comparison panel
    multi_signal_comparison()
    
    st.divider()
    
    # Individual signal analysis sections
    # Speech Analysis Section
    st.markdown('<h2 class="section-header">🎤 Speech Analysis</h2>', unsafe_allow_html=True)
    speech_file = st.file_uploader(
        "Upload Speech File (WAV, MP3)", 
        type=["wav", "mp3"],
        key="speech_uploader"
    )
    
    if speech_file:
        with st.spinner('Processing speech file...'):
            data, sr = process_audio(speech_file, "speech")
            if data is not None:
                st.session_state.processed_data['speech'] = (data, sr)
                st.success("Speech file processed successfully!")
                show_analysis("speech", data, sr)
    
    # Music Analysis Section
    st.markdown('<h2 class="section-header">🎵 Music Analysis</h2>', unsafe_allow_html=True)
    music_file = st.file_uploader(
        "Upload Music File (WAV, MP3)", 
        type=["wav", "mp3"],
        key="music_uploader"
    )
    
    if music_file:
        with st.spinner('Processing music file...'):
            data, sr = process_audio(music_file, "music")
            if data is not None:
                st.session_state.processed_data['music'] = (data, sr)
                st.success("Music file processed successfully!")
                show_analysis("music", data, sr)
    
    # EEG Analysis Section
    st.markdown('<h2 class="section-header">🧠 EEG Analysis</h2>', unsafe_allow_html=True)
    eeg_file = st.file_uploader(
        "Upload EEG Data (CSV)", 
        type=["csv"],
        key="eeg_uploader"
    )
    
    if eeg_file:
        with st.spinner('Processing EEG data...'):
            data, sr = process_eeg(eeg_file)
            if data is not None:
                st.session_state.processed_data['eeg'] = (data, sr)
                st.success("EEG data processed successfully!")
                show_analysis("eeg", data, sr)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: rgba(224, 230, 237, 0.7); padding: 20px;">
        <p><strong>Advanced Spectrogram Visualization System</strong></p>
        <p>Features: Time & Frequency Domain Analysis • Multi-Signal Comparison • Statistical Feature Extraction</p>
        <p>Created with Python, SciPy, Librosa, and Streamlit • © 2024 Signal Processing Lab</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()