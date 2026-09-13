import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from pillow_heif import register_heif_opener

# Register HEIF opener for iPhones
register_heif_opener()

st.set_page_config(page_title="Human Eye Detector Pro", page_icon="👁️", layout="wide")

st.title("👁️ Human Eye Detector Pro")

@st.cache_resource
def load_model():
    # Use "best.pt" for Cloud, or your full path for Local
    model_path = "best.pt" 
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

st.sidebar.header("Settings")
app_mode = st.sidebar.selectbox("Choose Mode", ["Image Upload", "Live Webcam"])

if app_mode == "Image Upload":
    st.subheader("📸 Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # --- PRE-PROCESSING FIX ---
        # We convert the image to RGB and resize it slightly before passing it to YOLO
        # This helps avoid "tiny eye" syndrome on high-res phone photos
        img_rgb = image.convert("RGB")
        img_array = np.array(img_rgb)
        
        col1, col2 = st.columns(2)

        with col1:
            st.header("Original")
            st.image(image, use_container_width=True)

        with st.spinner('Analyzing...'):
            # EXTREME CONFIDENCE: 0.05
            # We use a lower threshold to catch eyes in difficult lighting/resolutions
            results = model.predict(source=img_array, conf=0.05) 

            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.header("Detection Result")
            st.image(res_rgb, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Detection Analytics")
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                score = float(box.conf[0]) * 100
                st.write(f"**Eye {i+1}:** `{score:.2f}%` confidence")
            
            avg_conf = sum([float(b.conf[0]) for b in boxes]) / len(boxes)
            st.info(f"**Average Confidence:** `{avg_conf*100:.2f}%`")
        else:
            st.warning("No eyes detected. Tips: 1. Try a closer photo of the face. 2. Ensure good lighting. 3. Remove glasses if wearing them.")

elif app_mode == "Live Webcam":
    st.subheader("🎥 Live Tracking")
    run_webcam = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    if run_webcam:
        camera = cv2.VideoCapture(0)
        try:
            while run_webcam:
                ret, frame = camera.read()
                if not ret: break
                results = model.predict(source=frame, conf=0.05, verbose=False)
                res_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(res_rgb)
        finally:
            camera.release()
