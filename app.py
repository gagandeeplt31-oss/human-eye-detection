import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Human Eye Detector", 
    page_icon="👁️", 
    layout="wide"
)

st.title("👁️ Human Eye Detector")
st.markdown("""
    Welcome to the Eye Detection App! This app uses a custom-trained **YOLOv8** model 
    to detect human eyes and calculate the confidence level of each detection.
""")

# ==============================================================================
# 2. MODEL LOADING (Cached)
# ==============================================================================
@st.cache_resource
def load_model():
    # --- Ensure this path is correct for your local or cloud setup ---
    # For Cloud, use: model_path = "best.pt"
    # For Local, use the full path:
    model_path = "best.pt"
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model. Please check the path. Error: {e}")
    st.stop()

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.header("Application Settings")
app_mode = st.sidebar.selectbox("Choose Input Mode", ["Image Upload", "Live Webcam"])

st.sidebar.markdown("---")
st.sidebar.info("Built with YOLOv8 and Streamlit")

# ==============================================================================
# 4. MODE 1: IMAGE UPLOAD
# ==============================================================================
if app_mode == "Image Upload":
    st.subheader("📸 Upload Image Mode")
    st.write("Upload a photo to see the eyes and the AI's confidence levels.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.header("Original Image")
            st.image(image, use_container_width=True)

        with st.spinner('AI is analyzing...'):
            img_array = np.array(image)
            # Using conf=0.1 to be more forgiving and detect more eyes
            results = model.predict(source=img_array, conf=0.1) 

            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.header("Detection Result")
            st.image(res_rgb, use_container_width=True)
        
        # --- NEW: CONFIDENCE REPORT SECTION ---
        st.markdown("---")
        st.subheader("📊 Confidence Report")
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            # We create a list to store the confidence scores
            confidences = []
            for box in boxes:
                # box.conf[0] gives the confidence score as a decimal (e.g., 0.85)
                conf_score = float(box.conf[0]) * 100 
                confidences.append(conf_score)
            
            # Display each detection's confidence in a nice list
            for i, score in enumerate(confidences):
                st.write(f"**Eye {i+1}:** `{score:.2f}%` confidence")
            
            # Calculate and show the average confidence
            avg_conf = sum(confidences) / len(confidences)
            st.info(f"**Average Confidence:** `{avg_conf:.2f}%`")
        else:
            st.warning("No eyes were detected. Try a closer photo or different lighting.")

# ==============================================================================
# 5. MODE 2: LIVE WEBCAM
# ==============================================================================
elif app_mode == "Live Webcam":
    st.subheader("🎥 Live Webcam Tracking")
    st.write("This mode uses your local webcam to detect eyes in real-time.")
    
    run_webcam = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    if run_webcam:
        camera = cv2.VideoCapture(0)
        try:
            while run_webcam:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to access webcam.")
                    break

                # Using conf=0.1 for live tracking to prevent flickering
                results = model.predict(source=frame, conf=0.1, verbose=False)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(res_rgb)
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            camera.release()
            st.write("Webcam released.")
