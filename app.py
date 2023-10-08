# import os
# import tensorflow as tf
# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# import numpy as np

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def binary_classification_prediction(image_path, threshold=0.5):
#     try:
#         # Load the pre-trained model
#         model_path = 'brain_tumor_model.h5'  # Replace with the path to your model
#         model = tf.keras.models.load_model(model_path)

#         # Load and preprocess the image
#         img = image.load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)

#         # Make predictions
#         predictions = model.predict(x)

#         # Classify based on the threshold
#         if predictions[0][0] > threshold:
#             binary_class = 1  # Class 1
#             match_percentage = predictions[0][0] * 100
#         else:
#             binary_class = 0  # Class 0
#             match_percentage = (1 - predictions[0][0]) * 100

#         return binary_class, match_percentage

#     except Exception as e:
#         return None, None

# @app.route('/')
# def index():
#     return render_template('index.html', prediction_result=None)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return render_template('index.html', prediction_result="No file provided.")
    
#     file = request.files['file']

#     if file.filename == '':
#         return render_template('index.html', prediction_result="No selected file.")

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join('uploads', filename)
#         file.save(file_path)

#         # Perform binary classification prediction
#         binary_class, match_percentage = binary_classification_prediction(file_path)

#         if binary_class is not None:
#             if binary_class == 1:
#                 prediction_result = f"Tumor Detected (Match Percentage: {match_percentage:.2f}%)"
#             else:
#                 prediction_result = f"No Tumor Detected (Match Percentage: {match_percentage:.2f}%)"
#         else:
#             prediction_result = "Prediction failed. Please try again."

#         return render_template('index.html', prediction_result=prediction_result)
#     else:
#         return render_template('index.html', prediction_result="Invalid file format. Please upload a valid image.")

# if __name__ == '__main__':
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     app.run(debug=True)

import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

app = Flask(__name__)
model_path = 'models/brain_tumor_model.h5'  # Path to your brain tumor prediction model
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Save the uploaded image
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)

            # Load the pre-trained model
            model = tf.keras.models.load_model(model_path)

            # Load and preprocess the image
            img = image.load_img(image_path, target_size=(224, 224))
            print(image_path)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Make predictions
            predictions = model.predict(x)

            # Classify based on the threshold
            threshold = 0.5
            if predictions[0][0] > threshold:
                binary_class = 'Tumor'
                match_percentage = predictions[0][0] * 100
            else:
                binary_class = 'No Tumor'
                match_percentage = (1 - predictions[0][0]) * 100

            return render_template('index.html', result=(binary_class, round(match_percentage, 2), image_path))
        else:
            return render_template('index.html')

    return render_template('index.html', result=None)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

