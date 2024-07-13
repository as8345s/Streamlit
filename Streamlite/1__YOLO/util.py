import cv2 as cv
import streamlit as st
from ultralytics import YOLO
import numpy as np
import os

# Load a model with specific size.
def load_model(size: str = "Nano"):
    """Load model with given sizes [Nano, Small]

    :param size: Size of the model as string, currently available: Nano, Small
    :return: return the loaded model.
    """
    if size == "Nano":
        return YOLO("yolov8n.pt")
    elif size == 'Small':
        return YOLO("yolov8s.pt")
    else:
        return None


# Test if camera is available.
def test_cam(source: int = 0) -> bool:
    """Check if the source has a camera.

    :param source: Camera source as index, default is 0.
    :return: If a camera is available return True, else False.
    """
    cap = cv.VideoCapture(source)
    if cap is None or not cap.isOpened():
        return False
    return True


# Check if a video file exists.
def __check_recorded_video_file(path: str = ""):
    """Is there a video file?

    :param path: Path to the video file
    :return: True, if a video file exists.
    """
    return os.path.isfile(path)


# Create a placeholder frame.
def __placeholder_frame(text: str = "Placeholder", fontsize: int = 1, thick: int = 1):
    """Creat an empty frame to use as placeholder.

    :param text: Text to display like "No Video"
    :param fontsize: Size of the text to display.
    :param thick:    Thickness of the text to display.
    :return: returns a created frame.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv.putText(frame, text, (150, 240), cv.FONT_HERSHEY_SIMPLEX, fontScale=fontsize, color=(255, 255, 255), thickness=thick)
    return frame



# This function creates two video streams.
def __run_feed(placeholder1, placeholder2, cam:bool=False, model_size: str = 'Nano',  conf: float = 0.6):
    """

    :param placeholder1: Placeholder column 1.
    :param placeholder2: Placeholder column 2.
    :param cam:  Check if camera is on or off.
    :param model_size:  The chosen model size [Nano, Small]
    :param conf: float %, when to mark a class as positive.
           For example dog-object = 0.5, if cof==0.45 then mark this as dog.
    :return: Returns nothing.
    """

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
    placeholder1.image(__placeholder_frame('No camera activity', 1, 2))
    placeholder2.image(__placeholder_frame('No frames to predict', 1, 2))


# Function to create two video feeds side by side
def display_feed(cam_toggle: bool = False, model_size: str = 'Nano', cam_source: int = 0, conf: float = 0.6):
    """Creates two columns with placeholder to display video streams.

    :param cam_toggle: Check if camera is on or off.
    :param model_size: The chosen model size [Nano, Small]
    :param cam_source: The source of a camera as index.
    :param conf: conf: float %, when to mark a class as positive.
    :return: Returns nothing.
    """

    cam_available = test_cam(cam_source)

    col1, col2 = st.columns(2)

    # Placeholders for the video feeds
    with col1:
        st.write("Original")
        placeholder1 = st.empty()
    with col2:
        st.write("YOLOv8 Object Detection")
        placeholder2 = st.empty()

    if cam_available:
        __run_feed(placeholder1, placeholder2, cam_toggle, model_size, conf)
    else:
        placeholder1.image(__placeholder_frame('No camera activity', 1, 2))
        placeholder2.image(__placeholder_frame('No frames to predict', 1, 2))
        st.write("OOps.. No camera found at source ", cam_source)



# Function which shows a video stream and saves it.
def record_show_save_feed(cam_toggle: bool = False, cam_source: int = 0):
    """Display video stream and save it locally.

    :param cam_toggle: Check if camera is on or off.
    :param cam_source: The source of a camera as index.
    :return: returns nothing.
    """

    ## Notice: ##
    # - A problem might occur for the codec H264.
    # - https://github.com/cisco/openh264/releases  Downloade and extraxt .dll files.
    # openh264_path = r'C:\....\dll_dir
    # os.environ['PATH'] = openh264_path + ';' + os.environ['PATH']

    placeholder = st.empty()

    fps = 30.0
    resolution = (640, 480)
    video_path = './data/record.mp4'

    cam_available = test_cam(cam_source)

    if cam_available and cam_toggle:

        fourcc = cv.VideoWriter_fourcc(*'avc1') # H264  avc1
        out = cv.VideoWriter(video_path, fourcc, fps, resolution)

        cap = cv.VideoCapture(0)
        while cap.isOpened() :
            ret, frame = cap.read()
            if not ret:
                st.write("No camera frames")
                break

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            placeholder.image(frame, channels='RGB')
            out.write(frame)

        cap.release()
        placeholder.image(__placeholder_frame("No camera activity", 1, 2), channels="BGR")

    else:
        placeholder.image(__placeholder_frame("No camera activity", 1, 2), channels="BGR")
        st.write("No camera activity or wrong source: ", cam_source)


# Function to uploade a file in Streamlit.
def uploade_file(uploaded_file, cam_on_off: bool = False):
    """Uploade File.
    Uploade a local video file to the directory "data/record.mp4".
    It must be a .mp4 file and not larger than 100MB.

    :param uploaded_file: Uploaded File.
    :param cam_on_off: Check if camera is on or off.
    :return: returns nothing.
    """

    if cam_on_off:
        st.error("Please close the camera before uploading the video")
        return None

    if uploaded_file is not None:

        if uploaded_file.name != "record.mp4":
            st.error("Please name the file record.mp4, internally it will look for this file to make predictions")
            return None

        if uploaded_file.type == "video/mp4" and uploaded_file.size <= 100 * 1024 * 1024:
            file_path = os.path.join('./data', uploaded_file.name)

            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Video saved successfully: {file_path}")
        else:
            st.error("Invalid file format or size. Please upload a .mp4 file not larger than 100MB.")


# Load a video from the data directory to run prediction on it.
def run_inference(model_size: str = 'Nano'):
    """This function uses a YOLO model to run inference on a given video.
    The result is saved locally and is displayed in Streamlit.

    :param model_size: Size of the model as string [Nano, Small]
    :return: Returns nothing.
    """
    if __check_recorded_video_file('data/record.mp4'):

        st.write("Started processing, please wait for the success message. Check the Terminal.")

        model = load_model(model_size)

        video_input = './data/record.mp4'
        video_output = './data/result.mp4'
        vid = cv.VideoCapture(video_input)
        resolution = (640, 480)
        fps = 30.0

        # VideoWriter Objekt.
        fourcc = cv.VideoWriter_fourcc(*'H264')
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


# Displays Video or placeholder.
def display_inference_video():
    """If no video is found, then show a placeholder frame.
    After a video is uploaded or recorded, it will display it.

    :return: Returns nothing.
    """

    placeholder      = st.empty()
    novid_placeholder= __placeholder_frame('No video available', 1, 2)

    if os.path.isfile('data/result.mp4'):
        placeholder.video("data/result.mp4")
    else:
        placeholder.image(novid_placeholder, channels="BGR")
