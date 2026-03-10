import os
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template
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
if not os.path.exists(MODEL_PATH):
    print(f"Warning: {MODEL_PATH} not found. Please run train.py first.")
    session = None
else:
    session = ort.InferenceSession(MODEL_PATH)

def preprocess_image(image_bytes):
    # 1. Load Image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 2. Resize to 32x32 (CIFAR-100 size)
    img = img.resize((32, 32))
    
    # 3. Convert to NumPy and Normalize
    # Standard CIFAR-100 normalization values used in train.py
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = (img_data - mean) / std
    
    # 4. Transpose to (C, H, W) and add Batch Dimension
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({"error": "Model not loaded. Ensure model.onnx exists."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Preprocess
        img_bytes = file.read()
        input_data = preprocess_image(img_bytes)
        
        # Run Inference
        inputs = {session.get_inputs()[0].name: input_data}
        outputs = session.run(None, inputs)
        
        # Post-process
        prediction_idx = int(np.argmax(outputs[0]))
        class_name = CIFAR100_CLASSES[prediction_idx]
        confidence = float(np.max(outputs[0])) # Raw score (not softmax)

        return jsonify({
            "class_id": prediction_idx,
            "class_name": class_name,
            "confidence_score": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
