import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from werkzeug.utils import secure_filename

# Constants
MODEL_PATH = 'model_name.h5'  # Replace with the actual path to your .h5 model file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Allowed image formats
UPLOAD_FOLDER = 'uploads'  # Folder to save uploaded images

# Try loading the model with error handling
try:
    st.write("Loading model...")
    model = load_model(MODEL_PATH)
    st.write("Model loaded successfully!")
except OSError as e:
    st.error(f"OS Error while loading the model file: {e}")
    model = None
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    model = None

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Streamlit App UI
st.title("Pneumonia Detection Model")
st.write("Upload an image to check if pneumonia is detected.")

# File upload section
uploaded_file = st.file_uploader("Choose an image...", type=list(ALLOWED_EXTENSIONS))

if model is not None and uploaded_file is not None and allowed_file(uploaded_file.name):
    # Save the uploaded file
    filename = secure_filename(uploaded_file.name)
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Ensure the uploads directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Write the uploaded file to the designated folder
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and preprocess the image
    try:
        img = load_img(file_path, target_size=(224, 224))  # Adjust target size as needed
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize if required by the model

        # Predict using the model
        prediction = model.predict(img_array)
        result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "No Pneumonia Detected"

        # Display uploaded image and prediction
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write(f"Prediction: {result}")

        # Clean up by deleting the uploaded file after prediction (optional)
        os.remove(file_path)
    except Exception as e:
        st.error(f"An error occurred during image processing or prediction: {e}")

elif model is None:
    st.write("Model could not be loaded. Check the error messages above.")
elif uploaded_file is None:
    st.write("Please upload an image file.")
