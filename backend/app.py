import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from PIL import Image
import io

# Absolute path to this script's directory — works regardless of launch location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app) # Allows your React frontend to talk to this API

# --- 1. RECREATE THE MODEL ARCHITECTURE ---
# PyTorch needs to know the "shape" of the model to load the weights into
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# --- 2. LOAD THE TRAINED WEIGHTS ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading model on: {device}")

# 26 classes for A-Z
model = SignLanguageCNN(num_classes=26) 
model_path = os.path.join(BASE_DIR, 'msl_alphabet_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval() # STRICTLY evaluation mode (no learning)

# --- WARMUP: run one dummy inference to pre-compile MPS GPU shaders ---
# Without this, the very FIRST real request takes 30-90s on Apple Silicon.
print("Warming up model (compiling GPU shaders)...")
with torch.no_grad():
    dummy = torch.zeros(1, 3, 128, 128).to(device)
    model(dummy)
print("Model warmed up and ready.")

# Define the classes A-Z
class_names = [chr(i) for i in range(65, 91)] 

# Define how incoming images should be processed (must match training!)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# --- 3. CREATE THE API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    try:
        # Read and validate the image file size
        image_bytes = file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms and add batch dimension [1, 3, 128, 128]
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Make the prediction
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs, 1)

        # Calculate confidence from softmax probabilities
        probabilities = F.softmax(outputs, dim=1)
        confidence = round(probabilities[0][predicted_idx].item() * 100, 1)

        predicted_letter = class_names[predicted_idx.item()]
        
        return jsonify({
            'prediction': predicted_letter,
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting MSL Prediction Server...")
    # use_reloader=False prevents Flask from spawning a second watcher process.
    # Without this, the model loads and warms up TWICE on every startup.
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)