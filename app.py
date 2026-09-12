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
    to detect human eyes in real-time.
""")

# ==============================================================================
# 2. MODEL LOADING (Cached)
# ==============================================================================
@st.cache_resource
def load_model():
    # --- IMPORTANT: Ensure this path is exactly where your best.pt is located ---
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
    st.write("Upload a photo of a person, and the AI will draw boxes around the eyes.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        
        # Create two columns for a nice side-by-side view
        col1, col2 = st.columns(2)

        with col1:
            st.header("Original Image")
            st.image(image, use_container_width=True)

        # Run prediction
        with st.spinner('AI is analyzing...'):
            # Convert PIL Image to numpy array for YOLO
            img_array = np.array(image)
            results = model.predict(source=img_array, conf=0.25) 

            # Plot results (YOLO returns BGR by default)
            res_plotted = results[0].plot()
            # Convert BGR to RGB for Streamlit
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.header("Detection Result")
            st.image(res_rgb, use_container_width=True)
        
        # Display results summary
        num_eyes = len(results[0].boxes)
        if num_eyes > 0:
            st.success(f"Success! Found {num_eyes} eye(s) in the image.")
        else:
            st.warning("No eyes were detected in this image.")

# ==============================================================================
# 5. MODE 2: LIVE WEBCAM
# ==============================================================================
elif app_mode == "Live Webcam":
    st.subheader("🎥 Live Webcam Tracking")
    st.write("This mode uses your local webcam to detect eyes in real-time.")
    
    # Checkbox to start/stop the camera
    run_webcam = st.checkbox('Start Webcam')
    
    # Create an empty placeholder for the video feed
    # This prevents the app from creating a new image on every frame
    FRAME_WINDOW = st.image([])

    if run_webcam:
        # Initialize the camera
        camera = cv2.VideoCapture(0)
        
        try:
            while run_webcam:
                # Read a frame from the camera
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to access webcam. Please check your connection.")
                    break

                # Run prediction (verbose=False stops the console from filling with logs)
                results = model.predict(source=frame, conf=0.25, verbose=False)
                
                # Draw the boxes on the frame
                res_plotted = results[0].plot()
                
                # Convert BGR to RGB
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Update the placeholder with the new frame
                FRAME_WINDOW.image(res_rgb)
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        
        finally:
            # IMPORTANT: Always release the camera hardware when the loop ends
            camera.release()
            st.write("Webcam released.")

