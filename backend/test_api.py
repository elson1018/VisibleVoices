import requests

# 1. Point this to ANY image inside your Kaggle dataset
# e.g., '../msl_dataset/AlphabetsV2/AlphabetsV2/A/some_image_name.jpg'
image_path = 'test_image.png' 

# 2. The URL of your local Flask server
url = 'http://127.0.0.1:5001/predict'

print(f"Sending image to AI at {url}...")

try:
    # 3. Open the image and send it as a POST request
    with open(image_path, 'rb') as img:
        # 'file' is the exact key our Flask app is looking for: request.files['file']
        files = {'file': img} 
        response = requests.post(url, files=files)
    
    # 4. Print the AI's response!
    print("Response Code:", response.status_code)
    print("AI Prediction:", response.json())

except FileNotFoundError:
    print(f"Error: Could not find an image at {image_path}")