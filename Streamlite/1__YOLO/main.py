"""
Written in: IDE PyCharm.

"""


import streamlit as st
import util
import cv2

def main():

    st.title("YOLOv8 Object Detection")

    st.sidebar.title("Choose Model")
    st.sidebar.write("Size of thr YOLOv8 Model")
    model_size = st.sidebar.selectbox("Size", ["Nano", "Small"], key='modelsize')


    st.markdown("""
    Here you can use your own camera for **real time** object detection.\n
    We will be using a Pre-Trained model that was trained on the COCO Dataset.
    In this app we serve two different types of this model- nano with the ending n and a small with the ending s. 
    
    A requirement for it is a good CPU and a camera for sure.
    
    You can choose the size of the model in the sidebar. The default value is nano \n

    """)

    st.write("Selected model size:", model_size)

    cam = st.toggle("stop", value=False)
    st.write("cam: ", cam)

    # Start the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open the camera.")
        return

    stframe = st.empty()

    # Main loop to capture and display frames
    while cam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image. Please check your camera connection.")
            break

        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame)

    # Release the camera
    cap.release()






if __name__ == "__main__":
     main()