import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from werkzeug.utils import secure_filename

# Load the model once when the app starts
MODEL_PATH = 'model_name.h5'  # Make sure the model path is correct
model = load_model(MODEL_PATH)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Set up Streamlit UI
st.title("Pneumonia Detection Model")
st.write("Upload an image to check if pneumonia is detected or not.")

# File upload section
uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_EXTENSIONS)

if uploaded_file is not None:
    # Save the file temporarily
    filename = secure_filename(uploaded_file.name)
    file_path = os.path.join('uploads', filename)

    # Ensure the uploads directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the image from the saved file
    img = load_img(file_path, target_size=(224, 224))  # Adjust size if needed
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img /= 255.0  # Normalize the image if required

    # Predict the result
    prediction = model.predict(img)
    result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "No Pneumonia Detected"

    # Display the image and prediction
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: {result}")

    # Clean up by deleting the uploaded image after prediction (optional)
    os.remove(file_path)