from flask import Flask, request, render_template, redirect
import os
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
MODEL_PATH = 'model_name.h5'  # Make sure the model path is correct
model = load_model(MODEL_PATH)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)  # You can specify a path like 'uploads/'

        # Ensure the uploads directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the uploaded file
        file.save(file_path)

        # Load the image from the saved file
        img = load_img(file_path, target_size=(224, 224))  # Adjust size if needed
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img /= 255.0  # Normalize the image if required

        # Predict the result
        prediction = model.predict(img)
        result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "No Pneumonia Detected"

        # Clean up by deleting the uploaded image after prediction (optional)
        os.remove(file_path)

        return render_template('index.html', result=result)

    return "Invalid file type. Please upload a valid image."

if __name__ == '__main__':
    app.run(debug=True)
