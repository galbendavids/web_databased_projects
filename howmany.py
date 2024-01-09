import streamlit as st
from PIL import Image, ImageDraw
from mtcnn import MTCNN
import numpy as np
####

# Function to count the number of people in the image using mtcnn library
def count_people(image):
    # Load the image
    image_data = Image.open(image)
    image_array = image_data.convert("RGB")

    # Convert image to numpy array
    pixels = image_array
    pixels = pixels.convert('RGB')
    pixels = np.array(pixels)

    # Create the detector, using default weights
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(pixels)
    draw = ImageDraw.Draw(image_data)
    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, width, height = face['box']
        draw.rectangle(((x, y), (x + width, y + height)), outline=(0, 255, 0), width=2)

    # Count the number of people
    num_people = len(faces)

    return num_people

# Streamlit app
def main():
    st.title("People Counting App")
    st.sidebar.header("Upload Image")

    # Get user-uploaded image
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Check if the user has uploaded an image
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Check button
        if st.button("Check"):
            # Read the image
            image = uploaded_image
            image_data = Image.open(image)
            draw = ImageDraw.Draw(image_data)

            # Count people in the image
            num_people = count_people(image)

            # Display the result
            st.image(image_data, caption=f"Result - {num_people} people detected", use_column_width=True)
            st.write(f"Number of people detected: {num_people}")

if __name__ == "__main__":
    main()
