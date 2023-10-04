import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
Bootstrap(app)

# Define your model architecture
class YourModel(nn.Module):
    def __init__(self, num_classes):
        super(YourModel, self).__init__()
        # Define your model layers here (e.g., convolutional layers, fully connected layers)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 56 * 56, num_classes)  # Adjust input size based on your model
        
    def forward(self, x):
        # Define the forward pass of your model
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x

# Create an instance of your model
model = YourModel(num_classes=4)  # Adjust the number of classes as needed

# Load the saved model weights
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Function to preprocess the uploaded image
def preprocess_image(image):
    transform = torch.nn.Sequential(
        torch.nn.Resize((224, 224)),
        torch.nn.ToTensor(),
        torch.nn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # Check if the file has a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Check if the file is allowed
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not '.' in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
            return render_template('index.html', error='Invalid file type')

        # Read the image from the request
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        input_data = preprocess_image(image)

        # Perform inference using the loaded model
        with torch.no_grad():
            outputs = model(input_data)
            # Process the model's output to obtain predictions

        # Return the predictions
        predictions = ['Class 1', 'Class 2', 'Class 3']  # Replace with your actual predictions
        return render_template('index.html', predictions=predictions)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
