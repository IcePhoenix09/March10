import os
import numpy as np
import onnxruntime as ort
from flask import Flask, request, render_template
from PIL import Image
import io

app = Flask(__name__)

# CIFAR-100 Classes (Standard Order)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Load the ONNX model
MODEL_PATH = "model.onnx"
session = None
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((32, 32))
    
    # Use explicit float32 to avoid ONNX errors
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = (img_data - mean) / std
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    
    return img_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return render_template('index.html', error="Model not found. Run train.py first.")

    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    try:
        img_bytes = file.read()
        input_data = preprocess_image(img_bytes)
        
        inputs = {session.get_inputs()[0].name: input_data}
        outputs = session.run(None, inputs)
        
        prediction_idx = int(np.argmax(outputs[0]))
        class_name = CIFAR100_CLASSES[prediction_idx].replace('_', ' ')
        confidence = float(np.max(outputs[0]))

        return render_template('index.html', 
                               prediction=class_name, 
                               confidence=round(confidence, 2))

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
