"""
Written in: IDE PyCharm.

"""

import streamlit as st
import util

def main():

    st.title("YOLOv8 Object Detection")
    st.sidebar.title("Choose Model")
    st.sidebar.write("Size of thr YOLOv8 Model")
    model_size = st.sidebar.selectbox("Size", ["Nano", "Small"], key='modelsize')
    st.sidebar.write("Camera Source")
    cam_source = st.sidebar.selectbox("Source", [0, 1, 2, 3, 4, 5], key='src')
    st.sidebar.write('-------')
    st.sidebar.write("Mode parameter conf. % to mark a class as positive")
    conf = st.sidebar.slider("%:", 0.3, 0.9, step=0.05, value=0.6)

    st.markdown("""
    Object Detection without tracking*
    
    Here you can use your own camera for **real time** object detection.\n
    We will be using a Pre-Trained model that was trained on the COCO Dataset.
    In this app we serve two different types of this model- nano with the ending n and a small with the ending s. 
    
    A requirement for it is a good CPU and a camera for sure.
    
    You can choose the size of the model in the sidebar. The default value is nano \n

    """)

    st.write("Selected model size:", model_size)

    cam_on_off = st.toggle("Toggle camera (on, off)", value=False)

    util.display_feed(cam_on_off, model_size, cam_source, conf )


if __name__ == "__main__":
     main()