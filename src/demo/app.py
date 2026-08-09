import os
import sys
import tempfile
import streamlit as st
import time

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.inference.predict import predict_video

st.set_page_config(
    page_title="Shoplifting Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for a beautiful premium UI
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #00F0FF;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    h3 {
        color: #E0E0E0;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .metric-box {
        background: #1E2329;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        width: 250px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #FFF;
    }
    .metric-label {
        font-size: 14px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .shoplifting-red { color: #FF3B30 !important; }
    .normal-green { color: #34C759 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Smart Retail: Shoplifting Detection")
st.markdown("### Upload CCTV footage to detect anomalous behaviors instantly.")
st.write("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader("Upload a Video (.mp4)", type=["mp4"])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Run Detection 🚀", use_container_width=True):
            with st.spinner("Analyzing video frames using Deep Learning..."):
                # Save uploaded file temporarily to disk so OpenCV can read it
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                
                try:
                    start_time = time.time()
                    # Run inference using our predict_video pipeline
                    label, confidence = predict_video(tfile.name)
                    inf_time = time.time() - start_time
                    
                    # Determine color based on prediction
                    color_class = "shoplifting-red" if label == "Shoplifting" else "normal-green"
                    icon = "🚨" if label == "Shoplifting" else "✅"
                    
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-box">
                                <div class="metric-label">Prediction</div>
                                <div class="metric-value {color_class}">{icon} {label}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value {color_class}">{confidence:.2f}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Inference Time</div>
                                <div class="metric-value" style="color:#00F0FF;">{inf_time:.2f}s</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if label == "Shoplifting":
                        st.error("⚠️ ALERT: Suspicious activity detected! Security personnel should be notified.")
                    else:
                        st.success("✅ Clear: No suspicious activity detected in the footage.")
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                finally:
                    os.unlink(tfile.name)
