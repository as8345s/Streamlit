import cv2 as cv
import streamlit as st
from ultralytics import YOLO


def load_model(size: str = "Nano"):
    if size == "Nano":
        return YOLO("yolov8n.pt")
    elif size == 'Small':
        return YOLO("yolov8s.pt")
    else:
        return None


def test_cam(source: int = 0) -> bool:
    cap = cv.VideoCapture(source)
    if cap is None or not cap.isOpened():
        return False
    return True

def __run_feed(placeholder1, placeholder2, cam:bool=False, model_size: str = 'Nano',  conf: float = 0.6):

    model = load_model(model_size)

    cap = cv.VideoCapture(0)
    while cap.isOpened() and cam:
        ret, frame = cap.read()
        if not ret:
            st.write("No camera frames")
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        placeholder1.image(frame, channels='RGB')

        prediction_on_image = model(frame, conf=conf)
        annotated_frame = prediction_on_image[0].plot()

        placeholder2.image(annotated_frame, channels='RGB')

    cap.release()


# Function to create two video feeds side by side
def display_feed(cam_toggle: bool = False, model_size: str = 'Nano', cam_source: int = 0, conf: float = 0.6):

    cam_available = test_cam(cam_source)

    if cam_available:

        col1, col2 = st.columns(2)

        # Placeholders for the video feeds
        with col1:
            st.write("Original")
            placeholder1 = st.empty()
        with col2:
            st.write("YOLOv8 Object Detection")
            placeholder2 = st.empty()

        __run_feed(placeholder1, placeholder2, cam_toggle, model_size, conf)
    else:
        st.write("OOps.. No camera found at source ", cam_source)





