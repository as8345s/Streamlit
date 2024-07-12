import cv2 as cv
import streamlit as st
from ultralytics import YOLO
import numpy as np
import os


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

def __check_recorded_video_file(path:str=""):
    return os.path.isfile(path)


def __placeholder_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv.putText(frame, 'No camera activity', (150, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return frame

def __placeholder_frame_OB():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv.putText(frame, 'No frames to predict', (150, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return frame


def __placeholder_inference_video():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv.putText(frame, 'No video available', (150, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return frame

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
    placeholder1.image(__placeholder_frame())
    placeholder2.image(__placeholder_frame_OB())


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



def record_show_save_feed(cam_toggle: bool = False, cam_source: int = 0):
    placeholder = st.empty()

    fps = 30.0
    resolution = (640, 480)
    video_path = './data/record.mp4'

    cam_available = test_cam(cam_source)

    if cam_available and cam_toggle:

        # Erstelle VideoWriter Objekt.
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(video_path, fourcc, fps, resolution)

        cap = cv.VideoCapture(0)
        while cap.isOpened() :
            ret, frame = cap.read()
            if not ret:
                st.write("No camera frames")
                break

            #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            placeholder.image(frame, channels='RGB')
            out.write(frame)

        cap.release()
        placeholder.image(__placeholder_frame(), channels="BGR")

    else:
        placeholder.image(__placeholder_frame(), channels="BGR")
        st.write("No camera activity or wrong source: ", cam_source)


def uploade_file(uploaded_file, cam_on_off:bool=False):

    if cam_on_off:
        st.error("Please close the camera before uploading the video")
        return None

    if uploaded_file is not None:

        if uploaded_file.name != "record.mp4":
            st.error("Please name the file record.mp4, internally it will look for this file to make predictions")

        if uploaded_file.type == "video/mp4" and uploaded_file.size <= 100 * 1024 * 1024:
            file_path = os.path.join('./data', uploaded_file.name)

            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Video saved successfully: {file_path}")
        else:
            st.error("Invalid file format or size. Please upload a .mp4 file not larger than 100MB.")



def run_inference(model_size:str='Nano'):
    if __check_recorded_video_file('data/record.mp4'):

        st.write("Started processing, please wait for the success message. Check the Terminal.")

        model = load_model(model_size)

        video_input = './data/record.mp4'
        video_output = './data/result.mp4'
        vid = cv.VideoCapture(video_input)
        resolution = (640, 480)
        fps = 30.0

        # VideoWriter Objekt.
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(video_output, fourcc, fps, resolution)

        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                prediction_on_image = model(frame)
                annotated_frame = prediction_on_image[0].plot()
                out.write(annotated_frame)
            else:
                break

        vid.release()
        out.release()

        st.success("Inference done.")

    else:
        st.error("No file")



def display_inference_video():

    placeholder      = st.empty()
    novid_placeholder= __placeholder_inference_video()

    if os.path.isfile('data/result.mp4'):
        placeholder.video("data/result.mp4")
    else:
        placeholder.image(novid_placeholder, channels="BGR")











