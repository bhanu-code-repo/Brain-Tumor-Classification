import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Function to perform binary classification prediction
def binary_classification_prediction(image_path, threshold=0.5):
    try:
        # Load the pre-trained model
        model_path = 'models/brain_tumor_model.h5'  # Replace with the path to your model
        model = tf.keras.models.load_model(model_path)

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make predictions
        predictions = model.predict(x)

        # Classify based on the threshold
        if predictions[0][0] > threshold:
            binary_class = 1  # Class 1 (Tumor Detected)
            match_percentage = predictions[0][0] * 100
        else:
            binary_class = 0  # Class 0 (No Tumor Detected)
            match_percentage = (1 - predictions[0][0]) * 100

        return 'Tumor Detected' if binary_class == 1 else 'No Tumor Detected', match_percentage

    except Exception as e:
        return None, None

def analyze_image(file_path):
    # Perform image analysis and matching percentage calculation here
    # Replace this with your actual image analysis code
    return "Image Analysis Result", 75  # Sample result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(filename)
        file.save(filename)
        analysis_result, match_percentage = binary_classification_prediction(filename)
        return jsonify({'analysis_result': analysis_result, 'match_percentage': round(match_percentage, 2)})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
