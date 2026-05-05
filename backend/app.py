import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # Allows your React frontend to talk to this API

# --- 1. RECREATE THE MODEL ARCHITECTURE ---
# PyTorch needs to know the "shape" of the model to load the weights into
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- 2. LOAD THE TRAINED WEIGHTS ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading model on: {device}")

# 26 classes for A-Z
model = SignLanguageCNN(num_classes=26) 
model.load_state_dict(torch.load('msl_alphabet_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval() # STRICTLY evaluation mode (no learning)

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

    try:
        # Read the image file sent from the frontend
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms and add batch dimension [1, 3, 128, 128]
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Make the prediction
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs, 1)
            
        predicted_letter = class_names[predicted_idx.item()]
        
        return jsonify({
            'prediction': predicted_letter,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting MSL Prediction Server...")
    app.run(host='0.0.0.0', port=5001, debug=True)