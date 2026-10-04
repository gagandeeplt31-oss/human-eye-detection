from ultralytics import YOLO
import cv2

# 1. Load the 'Nano' model. 
# 'yolov8n.pt' is the smallest, fastest version of YOLOv8.
# The first time you run this, it will automatically download the model file.
model = YOLO('yolov8n.pt') 

# 2. Run detection on a sample image from the internet
# We'll use a picture of a bus provided by the YOLO team.
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)

print("\n--- SUCCESS! ---")
print("If you see this, your environment is working perfectly.")
print("Check your folder for a directory called 'runs/detect/predict' to see the image with boxes around the people and the bus!")
