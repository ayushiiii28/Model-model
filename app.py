import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the model using pickle (ensure you've saved it as 'model.pkl')
with open('training_history.pkl', 'rb') as f:
    model = pickle.load(f)

# Verify that the model is loaded correctly
if not hasattr(model, 'predict'):
    st.error("The loaded object is not a valid Keras model.")
else:
    st.write("Model loaded successfully!")

# Define the image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    # Resize and normalize the image
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit layout
st.title("Pneumonia Detection Using Chest X-rays")
st.write("Upload a chest X-ray image to check if it indicates pneumonia or not.")

# Add custom HTML content for UI (optional)
st.markdown("""
    <div style="text-align:center;">
        <h3>Upload an X-ray image and the model will predict whether the patient has pneumonia or not.</h3>
    </div>
""", unsafe_allow_html=True)

# Image uploader for X-ray
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image and make a prediction
    processed_image = preprocess_image(image)
    
    # Get prediction
    prediction = model.predict(processed_image)

    # Assuming the model outputs a binary classification (0: No Pneumonia, 1: Pneumonia)
    if prediction[0] > 0.5:  # This threshold can be adjusted
        st.write("### Result: Pneumonia Detected!")
    else:
        st.write("### Result: No Pneumonia Detected!")
    
    # Show the probability score (optional)
    st.write(f"Prediction Probability: {prediction[0][0]*100:.2f}%")

# Add a footer (optional)
st.markdown("""
    <footer style="text-align:center; padding-top:20px;">
        <p>Made with ❤️ using Streamlit</p>
    </footer>
""", unsafe_allow_html=True)
