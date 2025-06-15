import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import librosa
import librosa.display
import plotly.graph_objects as go
from st_aggrid import AgGrid
import time
import io

# Configure page with dark theme
st.set_page_config(
    page_title="Spectrogram Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS styling (keep the same as before)
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
    
    /* Matplotlib dark theme */
    .dark-theme-plot {
        background-color: rgba(35, 44, 75, 0.8) !important;
        border-radius: 10px;
        padding: 10px;
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

# Helper functions with dark theme support
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
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check if we have at least one numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in EEG data")
            
        # Use the first numeric column for analysis
        data = df[numeric_cols[0]].values
        
        # For demo, use a default sampling rate if not provided
        return data, 250  # 250 Hz default for EEG
    except Exception as e:
        st.error(f"Error processing EEG file: {str(e)}")
        st.warning("Please ensure your EEG file is a CSV with at least one numeric column")
        return None, None

def plot_waveform(data, sr, title):
    """Plot waveform with dark theme styling"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#232c4b')
    time = np.arange(0, len(data)) / sr
    ax.plot(time, data, color='#667eea', linewidth=1.5)
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

def plot_spectrogram(data, sr, title, cmap='viridis'):
    """Plot spectrogram using STFT with dark theme"""
    plt.style.use('dark_background')
    nperseg = min(1024, len(data) // 10)
    f, t, Zxx = stft(data, fs=sr, nperseg=nperseg)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#232c4b')
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

def show_analysis(signal_type, data, sr):
    """Display analysis for a signal type with dark theme"""
    with st.container():
        st.markdown(f'<div class="analysis-container">', unsafe_allow_html=True)
        
        st.subheader(f"{signal_type.capitalize()} Analysis", anchor=False)
        
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
        
        st.markdown("### Interactive Visualization")
        with st.spinner('Creating interactive plot...'):
            fig = plot_interactive_spectrogram(data, sr)
            st.plotly_chart(fig, use_container_width=True)
        
        # Signal stats
        st.markdown("### Signal Statistics")
        stats = {
            "Duration": f"{len(data)/sr:.2f} seconds",
            "Sampling Rate": f"{sr} Hz",
            "Max Amplitude": f"{np.max(np.abs(data)):.4f}",
            "Min Amplitude": f"{np.min(np.abs(data)):.4f}",
            "Mean Amplitude": f"{np.mean(np.abs(data)):.6f}",
            "Standard Deviation": f"{np.std(data):.6f}"
        }
        
        # Custom metrics display
        cols = st.columns(3)
        for i, (key, value) in enumerate(stats.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">{key}</h4>
                    <p style="font-size: 1.2rem; margin: 0;">{value}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Main app with dark theme
def main():
    # Header with gradient
    st.markdown("""
    <div class="main-header">
        <h1>Spectrogram Visualization System</h1>
        <p>Analyze non-stationary signals using Short-Time Fourier Transform (STFT)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    cols = st.columns(3)
    features = [
        ("🎤", "Speech Analysis", "Analyze voice patterns and phonetics"),
        ("🎵", "Music Analysis", "Visualize musical notes and frequencies"),
        ("🧠", "EEG Analysis", "Examine brain wave patterns and rhythms")
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
    
    # Analysis sections
    st.divider()
    
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
        <p>Spectrogram Visualization System • Created with Python, SciPy, and Streamlit</p>
        <p>© 2023 Signal Processing Lab • All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()