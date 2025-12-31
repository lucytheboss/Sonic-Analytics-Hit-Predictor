import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests
from pathlib import Path
import librosa

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "popularity_prediction_model.pkl"
DATA_PATH = BASE_DIR / "data" / "final_data.csv"

MODEL_URL = (
    "https://github.com/lucytheboss/Sonic-Analytics-Hit-Predictor/blob/main/models/popularity_prediction_model.pkl"
)
#

# --- 1. SETUP & HELPER FUNCTIONS ---
st.set_page_config(page_title="Hit Predictor AI", page_icon="🎵", layout="wide")

@st.cache_resource
def load_resources():
    # --- MODEL ---
    if not MODEL_PATH.exists():
        with st.spinner("📦 Downloading ML model..."):
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            r = requests.get(MODEL_URL)
            if r.status_code != 200:
                st.error("❌ Failed to download model")
                st.stop()
            MODEL_PATH.write_bytes(r.content)

    pkg = joblib.load(MODEL_PATH)

    # --- DATA ---
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        st.warning("⚠️ final_data.csv not found — running in demo mode")
        df = pd.DataFrame()

    return pkg, df


def extract_audio_features(uploaded_file):
    """
    Extracts BPM, Brightness, and Rhythm Strength from an MP3/WAV file.
    """
    # Load audio (Librosa handles the decoding)
    y, sr = librosa.load(uploaded_file, duration=30) # Limit to 30s for speed
    
    # 1. BPM (Tempo)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)
    
    # 2. Brightness (Spectral Centroid)
    # This correlates with "timbre" (Higher = Brighter/More Treble)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(spectral_centroids))
    
    # 3. Rhythm Strength (Onset Strength)
    # A proxy for how "punchy" or rhythmic the track is (0.0 to 1.0 approx)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    rhythm = float(np.mean(onset_env))
    
    # Normalize Rhythm to 0-1 scale visually (heuristic)
    rhythm = min(rhythm, 2.0) / 2.0 
    
    return bpm, brightness, rhythm

# Load everything
pkg, df_data = load_resources()

model_pop = pkg["model"]
model_genre = pkg["genre_model"]
scaler = pkg["scaler"]
feat_cols = pkg["features"]

st.title("🎵 Sonic Analytics: Hit Predictor")
st.markdown("Analyze market trends or predict the success of your new demo.")

tab1, tab2 = st.tabs(["📊 Market Analyzer", "🎧 AI Demo Rater"])

# ==========================================
# TAB 1: MARKET ANALYZER (Existing Songs)
# ==========================================
with tab1:
    st.header("Analyze Existing Hits")
    
    col_search, col_stats = st.columns([1, 3])
    
    with col_search:
        search_mode = st.radio("Search By:", ["Artist", "Track Title"])
        
        selected_song = None
        
        if search_mode == "Artist":
            # 1. Select Artist
            artists = sorted(df_data['artist_name'].unique())
            artist_input = st.selectbox("Select Artist", artists)
            
            # 2. Filter Songs by that Artist
            artist_songs = df_data[df_data['artist_name'] == artist_input]
            song_input = st.selectbox("Select Song", artist_songs['track_name'].unique())
            
            if song_input:
                selected_song = artist_songs[artist_songs['track_name'] == song_input].iloc[0]
                
        else:
            # Search by Track directly
            titles = sorted(df_data['track_name'].unique())
            song_input = st.selectbox("Select Song Title", titles)
            if song_input:
                selected_song = df_data[df_data['track_name'] == song_input].iloc[0]
                
                
        # --- IMAGE DISPLAY LOGIC ---
        if selected_song is not None:
            st.divider()
            
            # 1. Album Cover (Main Image)
            # Use 'cover_art_url' as found in your CSV
            if 'cover_art_url' in df_data.columns and pd.notna(selected_song['cover_art_url']):
                st.image(selected_song['cover_art_url'], width='stretch', caption=selected_song['track_name'])
            
            # 2. Artist Image (Below Album)
            # 'artist_image_url' exists in your list, so this should work now
            if 'artist_image_url' in df_data.columns and pd.notna(selected_song['artist_image_url']):
                st.image(selected_song['artist_image_url'], width='stretch', caption=selected_song['artist_name'])
        
        
    with col_stats:
        if selected_song is not None:
            st.subheader(f"{selected_song['track_name']}")
            st.caption(f"By {selected_song['artist_name']}")
            
            # 1. Display Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Popularity", f"{selected_song['track_popularity']}/100")
            m2.metric("Genre", selected_song['genre'])
            m3.metric("BPM", int(selected_song['bpm']))
            
            # 2. Comparison Logic (Song vs Genre Average)
            genre = selected_song['genre']
            genre_avg = df_data[df_data['genre'] == genre][['bpm', 'brightness', 'rhythm_strength']].mean()
            
            # Prepare Radar Chart Data
            categories = ['BPM', 'Brightness', 'Rhythm Strength']
            
            # Normalize for visualization (Simple Max Scaling)
            # You might want to adjust these divisors based on your real data maxes
            max_vals = {'bpm': 200, 'brightness': 5000, 'rhythm_strength': 1.0}
            
            song_vals = [
                selected_song['bpm']/max_vals['bpm'], 
                selected_song['brightness']/max_vals['brightness'], 
                selected_song['rhythm_strength']/max_vals['rhythm_strength']
            ]
            
            avg_vals = [
                genre_avg['bpm']/max_vals['bpm'], 
                genre_avg['brightness']/max_vals['brightness'], 
                genre_avg['rhythm_strength']/max_vals['rhythm_strength']
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=song_vals, theta=categories, fill='toself', name='This Song', line_color='#1DB954'
            ))
            fig.add_trace(go.Scatterpolar(
                r=avg_vals, theta=categories, fill='toself', name=f'{genre} Average', line_color='gray', opacity=0.5
            ))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Sonic Fingerprint vs. Genre Average")
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            st.info(f"Analysis: This track is **{'Brighter' if song_vals[1] > avg_vals[1] else 'Darker'}** and has **{'Stronger' if song_vals[2] > avg_vals[2] else 'Weaker'}** rhythm than the average {genre} song.")


with tab2:
    st.header("Upload Demo for Analysis")
    st.markdown("Upload your audio file. Our AI will extract the sonic features, while you provide the context (Artist fame).")
    
    col_upload, col_context = st.columns([1, 1])
    
    # Defaults
    audio_bpm, audio_bright, audio_rhythm = 120.0, 3000.0, 0.5
    
    with col_upload:
        uploaded_file = st.file_uploader("Upload MP3/WAV", type=['mp3', 'wav'])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/mp3')
            with st.spinner("Extracting features with Librosa..."):
                try:
                    # Run Librosa Extraction
                    audio_bpm, audio_bright, audio_rhythm = extract_audio_features(uploaded_file)
                    st.success("Audio Features Extracted!")
                    st.write(f"**Detected BPM:** {int(audio_bpm)}")
                    st.write(f"**Detected Brightness:** {int(audio_bright)}")
                    st.write(f"**Detected Rhythm:** {audio_rhythm:.2f}")
                except Exception as e:
                    st.error(f"Error reading audio: {e}")

    with col_context:
        st.subheader("Market Context")
        st.markdown("*The model needs to know who is releasing this track.*")
        
        # User must provide these since the audio file doesn't have them
        artist_pop = st.slider("Artist Current Popularity", 0, 100, 50)
        prev_pop = st.slider("Previous Track Success", 0, 100, 50)

    # --- PREDICTION BUTTON ---
    # --- PREDICTION BUTTON ---
    if st.button("🔮 Predict Success"):
        if uploaded_file is None:
            st.warning("Please upload an audio file first.")
        else:
            # 1. Prepare Audio Data for Scaling
            # The scaler expects exactly these 5 columns in this order:
            # ['bpm', 'energy', 'brightness', 'noisiness', 'rhythm_strength']
            
            # Since we only extracted 3 features, we use averages/defaults for the others
            # (In a real app, you would extract these with Librosa too)
            defaults = {
                'energy': 0.6,      # Default mid-range energy
                'noisiness': 0.05   # Default low noisiness
            }
            
            # Create the DataFrame matching the scaler's training data
            audio_features_df = pd.DataFrame([[
                audio_bpm, 
                defaults['energy'], 
                audio_bright, 
                defaults['noisiness'], 
                audio_rhythm
            ]], columns=['bpm', 'energy', 'brightness', 'noisiness', 'rhythm_strength'])
            
            # 2. Scale the Audio Features
            try:
                scaled_audio = scaler.transform(audio_features_df)
                # scaled_audio is now a numpy array of shape (1, 5)
            except Exception as e:
                st.error(f"Scaling Error: {e}")
                st.stop()
            
            # 3. Predict Genre
            # The Genre Model uses these 5 scaled audio features
            pred_genre = model_genre.predict(scaled_audio)[0]
            
            # 4. Predict Popularity (Fix for 'const' error)
            
            # A. Create the Template Row based on your saved features
            input_row = pd.DataFrame(0, index=[0], columns=feat_cols)
            
            # B. Fill in the Data
            if 'prev_track_popularity' in input_row.columns: 
                input_row['prev_track_popularity'] = prev_pop
            
            if 'bpm' in input_row.columns: input_row['bpm'] = scaled_audio[0][0]
            if 'energy' in input_row.columns: input_row['energy'] = scaled_audio[0][1]
            if 'brightness' in input_row.columns: input_row['brightness'] = scaled_audio[0][2]
            if 'noisiness' in input_row.columns: input_row['noisiness'] = scaled_audio[0][3]
            if 'rhythm_strength' in input_row.columns: input_row['rhythm_strength'] = scaled_audio[0][4]
                
            # C. Fill the Genre Dummy
            if pred_genre in input_row.columns:
                input_row[pred_genre] = 1
            elif f"genre_{pred_genre}" in input_row.columns:
                input_row[f"genre_{pred_genre}"] = 1
            
            # --- CRITICAL FIX: DROP CONST ---
            # The error explicitly said 'const' was unseen, so we remove it.
            if 'const' in input_row.columns:
                input_row = input_row.drop(columns=['const'])
            
            # D. Run Prediction
            try:
                score = model_pop.predict(input_row)[0]
                score = np.clip(score, 0, 100)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                score = 0
            
            # 5. Display Results
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Genre", pred_genre)
            r2.metric("Success Score", f"{int(score)}/100")
            r3.metric("BPM", int(audio_bpm))
            
            # 6. Recommendation Engine
            st.subheader(f"👨‍⚕️ Song Doctor Recommendation")
            
            if 'genre_recipes' in pkg:
                recipe = pkg['genre_recipes'].get(pred_genre)
                if recipe:
                    target_bright = recipe.get('brightness', 3000)
                    target_rhythm = recipe.get('rhythm_strength', 0.5)
                    
                    advice = []
                    if audio_bright < target_bright * 0.8:
                        advice.append("🔊 **Too Dark:** Increase high-end frequencies (treble).")
                    elif audio_bright > target_bright * 1.2:
                        advice.append("🔉 **Too Bright:** Reduce harsh high frequencies.")
                        
                    if audio_rhythm < target_rhythm * 0.8:
                        advice.append("🥁 **Weak Rhythm:** Make the drums punchier.")
                    
                    if not advice:
                        st.success("✅ Audio features align perfectly with this genre!")
                    else:
                        for tip in advice:
                            st.warning(tip)