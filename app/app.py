import streamlit as st
import requests
import time

# Configure the page
st.set_page_config(page_title="AI Call Analyzer", page_icon="🎙️", layout="wide")

st.title("🎙️ AI Call Sentiment & Insights Dashboard")
st.markdown("Upload a customer call recording to extract sentiment, diarization metrics, and coaching insights.")

# --- Sidebar for File Upload ---
st.sidebar.header("1. Upload Call Audio")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
    
    if st.sidebar.button("Analyze Call", type="primary"):
        
        # UI Loading State
        with st.spinner("Transcribing and analyzing via FastAPI backend..."):
            
            try:
                # --- API Connection ---
                # This points to your local FastAPI server
                api_url = "http://localhost:8000/analyze-call"
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                
                # Send the POST request to your main.py endpoint
                response = requests.post(api_url, files=files)
                response.raise_for_status() # Check for HTTP errors
                
                data = response.json()
                
                st.toast('Analysis Complete!', icon='✅')

                # --- Dashboard Layout ---
                st.divider()
                
                # Row 1: High-Level Metrics
                col1, col2, col3 = st.columns(3)
                
                # Color code the sentiment
                sentiment_color = "green" if data["sentiment"] == "POSITIVE" else "red" if data["sentiment"] == "NEGATIVE" else "gray"
                
                col1.metric("Overall Sentiment", data["sentiment"])
                col2.metric("Questions Asked", data["metrics"]["num_questions"])
                col3.metric("Longest Monologue", f"{data['metrics']['longest_monologue_s']}s")

                st.divider()

                # Row 2: Actionable Insights & Talk Time
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader("💡 AI Coaching Insight")
                    st.info(data["metrics"]["insight"], icon="🤖")
                
                with col_right:
                    st.subheader("⏱️ Talk Time Ratio")
                    rep_time = data["metrics"]["talk_time_ratio"]["A"] * 100
                    client_time = data["metrics"]["talk_time_ratio"]["B"] * 100
                    
                    st.progress(int(rep_time), text=f"Speaker A: {rep_time:.1f}%")
                    st.progress(int(client_time), text=f"Speaker B: {client_time:.1f}%")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend. Is your FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    # Empty state
    st.info("👈 Please upload an audio file in the sidebar to begin analysis.")
