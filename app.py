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

st.set_page_config(page_title="Hit Predictor AI", page_icon="🎵", layout="wide")

@st.cache_resource
def load_resources():
    raw_model_url = (
        "https://github.com/lucytheboss/Sonic-Analytics-Hit-Predictor/raw/main/models/popularity_prediction_model.pkl"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        with st.spinner("📦 Downloading ML model..."):
            try:
                r = requests.get(raw_model_url)
                r.raise_for_status()
                MODEL_PATH.write_bytes(r.content)
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                st.stop()

    try:
        pkg = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Model corrupted. Deleting and retrying... Error: {e}")
        MODEL_PATH.unlink() 
        st.stop()

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        st.warning("⚠️ final_data.csv not found — running in demo mode")
        df = pd.DataFrame()

    return pkg, df


def extract_audio_features(uploaded_file):
    y, sr = librosa.load(uploaded_file, duration=None) 
    duration_sec = librosa.get_duration(y=y, sr=sr)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)
    
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(spectral_centroids))
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    rhythm = float(np.mean(onset_env))
    rhythm = min(rhythm, 2.0) / 2.0 
    
    # Time-Series Graph Data
    rms = librosa.feature.rms(y=y)[0]
    
    # Normalize
    times = librosa.times_like(rms, sr=sr)
    norm_rms = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    
    cent_aligned = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    min_len = min(len(times), len(cent_aligned))
    times = times[:min_len]
    norm_rms = norm_rms[:min_len]
    cent = cent_aligned[:min_len]
    norm_cent = (cent - np.min(cent)) / (np.max(cent) - np.min(cent))
    
    hop = int(len(times) / (duration_sec * 2)) if duration_sec > 0 else 1
    hop = max(1, hop)
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=times[::hop], y=norm_rms[::hop],
        mode='lines', name='Energy (Loudness)',
        line=dict(color='#1DB954', width=2),
        fill='tozeroy', fillcolor='rgba(29, 185, 84, 0.1)'
    ))
    
    fig_timeline.add_trace(go.Scatter(
        x=times[::hop], y=norm_cent[::hop],
        mode='lines', name='Brightness (Timbre)',
        line=dict(color='#1E90FF', width=2)
    ))
    
    fig_timeline.update_layout(
        title="📈 Feature Evolution: Energy & Brightness over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized Intensity (0-1)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.1)
    )

    return bpm, brightness, rhythm, duration_sec, fig_timeline


pkg, df_data = load_resources()

model_pop = pkg["model"]
model_genre = pkg["genre_model"]
scaler = pkg["scaler"]
feat_cols = pkg["features"]

st.title("🎵 Sonic Analytics: Hit Predictor")
st.markdown("Analyze market trends or predict the success of your new demo.")

tab1, tab2 = st.tabs(["📊 Market Analyzer", "🎧 AI Demo Rater"])


# TAB 1: MARKET ANALYZER (Existing Songs)
with tab1:
    st.header("Analyze Existing Hits")
    
    col_search, col_stats = st.columns([1, 3])
    
    with col_search:
        search_mode = st.radio("Search By:", ["Artist", "Track Title"])
        
        selected_song = None
        
        if search_mode == "Artist":

            artists = sorted(df_data['artist_name'].unique())
            artist_input = st.selectbox("Select Artist", artists)
            
            artist_songs = df_data[df_data['artist_name'] == artist_input]
            song_input = st.selectbox("Select Song", artist_songs['track_name'].unique())
            
            if song_input:
                selected_song = artist_songs[artist_songs['track_name'] == song_input].iloc[0]
                
        else:
            titles = sorted(df_data['track_name'].unique())
            song_input = st.selectbox("Select Song Title", titles)
            if song_input:
                selected_song = df_data[df_data['track_name'] == song_input].iloc[0]
                
                
        if selected_song is not None:
            st.divider()

            if 'cover_art_url' in df_data.columns and pd.notna(selected_song['cover_art_url']):
                st.image(selected_song['cover_art_url'], width='stretch', caption=selected_song['track_name'])
            
            if 'artist_image_url' in df_data.columns and pd.notna(selected_song['artist_image_url']):
                st.image(selected_song['artist_image_url'], width='stretch', caption=selected_song['artist_name'])
        
        
    with col_stats:
        if selected_song is not None:
            st.subheader(f"{selected_song['track_name']}")
            st.caption(f"By {selected_song['artist_name']}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Popularity", f"{selected_song['track_popularity']}/100")
            m2.metric("Genre", selected_song['genre'])
            m3.metric("BPM", int(selected_song['bpm']))
            
            genre = selected_song['genre']
            genre_avg = df_data[df_data['genre'] == genre][['bpm', 'brightness', 'rhythm_strength']].mean()
            
            categories = ['BPM', 'Brightness', 'Rhythm Strength']
            
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
            
            st.info(f"Analysis: This track is **{'Brighter' if song_vals[1] > avg_vals[1] else 'Darker'}** and has **{'Stronger' if song_vals[2] > avg_vals[2] else 'Weaker'}** rhythm than the average {genre} song.")


# TAB 2: AI DEMO RATER
with tab2:
    st.header("Upload Demo for Analysis")
    st.markdown("Upload your audio file. Our AI will extract the sonic features, while you provide the context.")
    
    col_upload, col_context = st.columns([1, 1])
    
    if 'audio_bpm' not in st.session_state: st.session_state['audio_bpm'] = 120.0
    if 'audio_bright' not in st.session_state: st.session_state['audio_bright'] = 3000.0
    if 'audio_rhythm' not in st.session_state: st.session_state['audio_rhythm'] = 0.5
    if 'audio_duration' not in st.session_state: st.session_state['audio_duration'] = 180.0
    if 'feature_fig' not in st.session_state: st.session_state['feature_fig'] = None
    if 'has_audio' not in st.session_state: st.session_state['has_audio'] = False
    
    with col_upload:
        uploaded_file = st.file_uploader("Upload MP3/WAV", type=['mp3', 'wav'])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/mp3')
            
            if st.button("🔍 Extract Audio Features"):
                with st.spinner("Analyzing full track structure (this may take a moment)..."):
                    try:
                        bpm, bright, rhythm, dur, fig = extract_audio_features(uploaded_file)
                        
                        st.session_state['audio_bpm'] = bpm
                        st.session_state['audio_bright'] = bright
                        st.session_state['audio_rhythm'] = rhythm
                        st.session_state['audio_duration'] = dur
                        st.session_state['feature_fig'] = fig
                        st.session_state['has_audio'] = True
                        
                        st.success("Features Extracted Successfully!")
                    except Exception as e:
                        st.error(f"Error reading audio: {e}")

            if st.session_state['has_audio']:
                mins = int(st.session_state['audio_duration'] // 60)
                secs = int(st.session_state['audio_duration'] % 60)
                
                c1, c2 = st.columns(2)
                c1.write(f"**BPM:** {int(st.session_state['audio_bpm'])}")
                c1.write(f"**Brightness:** {int(st.session_state['audio_bright'])}")
                c2.write(f"**Rhythm:** {st.session_state['audio_rhythm']:.2f}")
                c2.write(f"**Duration:** {mins}:{secs:02d}")
                
    with col_context:
        st.subheader("Market Context")
        st.markdown("*The model needs to know who is releasing this track.*")
        artist_pop = st.slider("Artist Current Popularity", 0, 100, 50)
        prev_pop = st.slider("Previous Track Success", 0, 100, 50)

    if st.session_state['has_audio'] and st.session_state['feature_fig']:
        st.divider()
        st.plotly_chart(st.session_state['feature_fig'], use_container_width=True)

    if st.button("🔮 Predict Success"):
        if not st.session_state['has_audio']:
            st.warning("Please upload audio and click 'Extract Audio Features' first.")
        else:
            bpm = st.session_state['audio_bpm']
            bright = st.session_state['audio_bright']
            rhythm = st.session_state['audio_rhythm']
            duration_sec = st.session_state['audio_duration']
            
            defaults = {'energy': 0.6, 'noisiness': 0.05}
            
            audio_features_df = pd.DataFrame([[
                bpm, defaults['energy'], bright, defaults['noisiness'], rhythm
            ]], columns=['bpm', 'energy', 'brightness', 'noisiness', 'rhythm_strength'])
            
            try:
                scaled_audio = scaler.transform(audio_features_df)
                pred_genre = model_genre.predict(scaled_audio)[0]
            except Exception as e:
                st.error(f"Model Error: {e}")
                st.stop()
            
            input_row = pd.DataFrame(0, index=[0], columns=feat_cols)
            
            if 'prev_track_popularity' in input_row.columns: 
                input_row['prev_track_popularity'] = prev_pop
            
            cols_map = {
                'bpm': scaled_audio[0][0], 
                'energy': scaled_audio[0][1],
                'brightness': scaled_audio[0][2], 
                'noisiness': scaled_audio[0][3],
                'rhythm_strength': scaled_audio[0][4]
            }
            for col, val in cols_map.items():
                if col in input_row.columns: input_row[col] = val
                
            if pred_genre in input_row.columns:
                input_row[pred_genre] = 1
            elif f"genre_{pred_genre}" in input_row.columns:
                input_row[f"genre_{pred_genre}"] = 1
            
            if 'const' in input_row.columns: 
                input_row = input_row.drop(columns=['const'])
            
            score = model_pop.predict(input_row)[0]
            score = np.clip(score, 0, 100)
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Genre", pred_genre)
            c2.metric("Success Score", f"{int(score)}/100")
            c3.metric("Duration", f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}")
            
            st.subheader(f"👨‍⚕️ Song Doctor: {pred_genre} Analysis")
            
            default_recipes = {
                'Alternative':       {'bpm': 112, 'brightness': 2102, 'rhythm_strength': 0.66},
                'Christian':         {'bpm': 111, 'brightness': 2428, 'rhythm_strength': 0.70},
                'Christmas: Pop':    {'bpm': 124, 'brightness': 2858, 'rhythm_strength': 0.81},
                'Country':           {'bpm': 118, 'brightness': 2165, 'rhythm_strength': 0.67},
                'Dance':             {'bpm': 124, 'brightness': 2648, 'rhythm_strength': 0.71},
                'Hard Rock':         {'bpm': 119, 'brightness': 2581, 'rhythm_strength': 0.62},
                'Hip-Hop/Rap':       {'bpm': 115, 'brightness': 2331, 'rhythm_strength': 0.88},
                'Holiday':           {'bpm': 111, 'brightness': 2005, 'rhythm_strength': 0.64},
                'Metal':             {'bpm': 119, 'brightness': 2761, 'rhythm_strength': 0.59},
                'Pop':               {'bpm': 117, 'brightness': 2444, 'rhythm_strength': 0.76},
                'R&B/Soul':          {'bpm': 120, 'brightness': 2305, 'rhythm_strength': 0.76},
                'Rock':              {'bpm': 120, 'brightness': 2216, 'rhythm_strength': 0.67},
                'Singer/Songwriter': {'bpm': 108, 'brightness': 2152, 'rhythm_strength': 0.71},
                'Soundtrack':        {'bpm': 118, 'brightness': 2381, 'rhythm_strength': 0.74},
            }
            
            target_bpm = 120
            target_bright = 3000
            target_rhythm = 0.5
            target_duration = 210000
            source_type = "Baseline"
            
            if pred_genre in default_recipes:
                recipe = default_recipes[pred_genre]
                target_bpm = recipe['bpm']
                target_bright = recipe['brightness']
                target_rhythm = recipe['rhythm_strength']
                source_type = "Ideal Recipe"

            if not df_data.empty and 'duration_ms' in df_data.columns:
                if pred_genre in df_data['genre'].values:
                    target_duration = df_data[df_data['genre'] == pred_genre]['duration_ms'].mean()

            st.caption(f"Comparing against: {source_type} for {pred_genre}")

            norm_max = {'bpm': 200, 'bright': 5000, 'rhythm': 1.0}
            
            fig_doc = go.Figure()
            
            fig_doc.add_trace(go.Scatterpolar(
                r=[bpm/norm_max['bpm'], bright/norm_max['bright'], rhythm/norm_max['rhythm']],
                theta=['BPM', 'Brightness', 'Rhythm'],
                fill='toself', name='Your Demo', line_color="#1DB954"
            ))
            
            fig_doc.add_trace(go.Scatterpolar(
                r=[target_bpm/norm_max['bpm'], target_bright/norm_max['bright'], target_rhythm/norm_max['rhythm']],
                theta=['BPM', 'Brightness', 'Rhythm'],
                fill='toself', name=f'{pred_genre} Target', line_color="#D01BBE", opacity=0.6
            ))
            
            fig_doc.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=400, title="Sonic Fingerprint vs. Target")
            st.plotly_chart(fig_doc, use_container_width=True)
            
            advice = []
            
            duration_ms = duration_sec * 1000
            dur_diff = duration_ms - target_duration
            
            if dur_diff > 45000:
                advice.append(f"⏱️ **Too Long:** Your track is {int(dur_diff/1000)}s longer than the {pred_genre} average. Consider a 'Radio Edit' for streaming.")
            elif dur_diff < -45000:
                advice.append(f"⏱️ **Too Short:** This is significantly shorter than the {pred_genre} average. Ensure the arrangement feels complete.")
            
            if bright < target_bright * 0.85:
                advice.append(f"🔉 **Too Dark:** Your track is muddy compared to typical {pred_genre}. Boost high-end EQ (Treble).")
            elif bright > target_bright * 1.15:
                advice.append(f"🔊 **Too Bright:** Your track is harsh. Try cutting some high frequencies or using warmer instrumentation.")
                
            if rhythm < target_rhythm * 0.85:
                advice.append(f"🥁 **Low Energy:** Rhythm strength is weak for {pred_genre}. Make the drums punchier or add percussion.")
            elif rhythm > target_rhythm * 1.25:
                advice.append(f"💥 **Too Aggressive:** Rhythm is very intense. Ensure it fits the {pred_genre} vibe.")
                
            if not advice:
                st.balloons()
                st.success(f"✅ **Perfect Fit!** Your demo hits the {pred_genre} sonic sweet spot perfectly!")
            else:
                st.warning("⚠️ **Doctor's Orders:**")
                for tip in advice:
                    st.write(tip)