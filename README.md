# 🎵 Sonic Analytics: Music Hit Predictor
**A Full-Stack Data Science & Machine Learning Application**

Sonic Analytics is an intelligent music analysis tool that leverages **Audio Signal Processing (DSP)** and **Machine Learning** to predict the potential success of a song. By combining metadata from Apple Music with popularity metrics from Spotify and raw audio analysis via Librosa, this app provides deep insights into market trends and song characteristics.

## 🚀 Key Features
1. **📈 Market Explorer**
    - **Cross-Platform Data**: Merges Apple Music library data with Spotify Popularity metrics.
    - **Sonic Fingerprinting**: Visualizes the "DNA" of a song (Energy, BPM, Brightness) using Radar Charts.
    - **Artist Insights**: Displays Artist Popularity and Follower counts to contextualize track success.

2. **🤖 The "Hit Predictor" **AI**
    - **Success Regression**: A Random Forest Regressor that predicts a track's potential "Popularity Score" (0-100) based on its audio features.
    - **Genre Classifier**: AI model that listens to an MP3 and predicts its genre with confidence intervals.
    - **Detailed Audio Analysis**: Extracts 5 key features from raw audio:
        - BPM (Tempo)
        - Energy (RMS)
        - Brightness (Spectral Centroid)
        - Noisiness (Zero Crossing Rate)
        - Rhythm Stability

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Data Collection**: `spotipy` (Spotify Web API), `requests`
- **Data Processing**: `pandas`, `numpy`
- **Audio Analysis**: `librosa` (Digital Signal Processing)
- **Machine Learning**: `scikit-learn` (Random Forest, Classification & Regression)
- **Web App**: `streamlit`, `plotly` (Interactive Visualization)

## ⚙️ Architecture Pipeline
- Data Ingestion: Raw dataset from Apple Music (Track Name, Artist).
- Enrichment: Python script searches Spotify API to append Popularity Scores, Cover Art, and Artist Metadata.

Audio Extraction: Librosa processes 30-second preview clips to extract mathematical audio features.

Model Training:

Input: Audio Features (BPM, Energy, Brightness, etc.)

Target: Spotify Popularity Score.

Deployment: Streamlit app loads the trained models (.pkl) to serve real-time predictions on user uploads.