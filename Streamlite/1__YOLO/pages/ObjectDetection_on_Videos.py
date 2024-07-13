"""
Written in: IDE PyCharm.
Autor:      Alex.S
use.:
    This page allows users to use their camera to record a video or alternatively uploade a video
    to run inference on it with a YOLO Model for Object Detection.

    All recorded videos are saved locally.
"""


import streamlit as st
import util
import asyncio

def page():

    st.title("Object Detection on given videos")

    st.sidebar.write("Camera Source")
    cam_source = st.sidebar.selectbox("Source", [0, 1, 2, 3, 4, 5], key='src')
    st.sidebar.write("Size of the YOLOv8 Model")
    model_size = st.sidebar.selectbox("Size", ["Nano", "Small"], key='modelsize')

    st.markdown("""
    Object Detection without tracking*
   
    Here you can use your own camera to record and save a video.
    If you don't have a camera, you can uploade a video by using the uploader down there.
   
    The recorded video will be saved in the directory data/record.mp4

     """)

    st.write("Record and save video.:")
    # Turn on / off camera.
    cam_on_off = st.toggle("Toggle camera to record and save", value=False)
    # Use camera and display video on the page, the stream is being saved.
    util.record_show_save_feed(cam_on_off, cam_source)


    st.write("Here you can uploade a video file. Name the video file 'record.mp4'. The file size "
             "should not be larger than 100MB.\n"
             "Turn **off** your camera before you uploade a video.")

    # Function to uploade a video file, format: .mp4
    uploaded_file = st.file_uploader("Choose a video", type=["mp4"])
    util.uploade_file(uploaded_file, cam_on_off)

    st.write("Now you can run inference on the given Video. You can select the Model-Type and "
             "other parameters. At the end you can watch the Video, it will be saved as result.mp4 at data/ ")

    # Button to trigger a function to run inference on a given video.
    if st.button("Run inference.", key="start_inference"):
        util.run_inference(model_size)

    # Display the inference result.
    util.display_inference_video()


page()
