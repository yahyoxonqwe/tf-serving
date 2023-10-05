import cv2 
import tempfile
import time
from tf_utils import detect_image
import streamlit as st
from PIL import Image

# Streamlit app
st.title("Yolov8 object detection")

# Upload an image
def main():
# Create a video file uploader
    video_path = None
    st.header("Upload a video for inference")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])


    # If the user chooses to use the default video

    # If the user chooses to use the uploaded video
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    # If there's a video to process, do the inference
    if video_path is not None:
        # Load the video with cv2
        cap = cv2.VideoCapture(video_path)
        for_fps = st.empty()
        outputing = st.empty()

        fps = 0
        prev_time = 0
        curr_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run the inference
            output = detect_image(frame)

            # Convert the output to an image that can be displayed
            output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

            # Display the image
            outputing.image(output_image)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            for_fps.write(f"FPS: {fps}")
            # print(fps)
        cap.release()
    else:
        st.write("Please upload a video file .")

if __name__ == "__main__":
    main()