import os
import numpy as np
import onnxruntime as ort
from flask import Flask, request, render_template, redirect, url_for, session
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = "secret_key"

# CIFAR-100 Classes
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
ort_session = None
if os.path.exists(MODEL_PATH):
    ort_session = ort.InferenceSession(MODEL_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((32, 32))
    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
    std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = (img_data - mean) / std
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    return img_data

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if ort_session is None:
            session['error'] = "Model not found."
            return redirect(url_for('home'))

        if 'file' not in request.files:
            session['error'] = "No file uploaded."
            return redirect(url_for('home'))
        
        file = request.files['file']
        if file.filename == '':
            session['error'] = "No file selected."
            return redirect(url_for('home'))

        try:
            img_bytes = file.read()
            input_data = preprocess_image(img_bytes)
            inputs = {ort_session.get_inputs()[0].name: input_data}
            outputs = ort_session.run(None, inputs)
            prediction_idx = int(np.argmax(outputs[0]))
            
            # Store results in session
            session['prediction'] = CIFAR100_CLASSES[prediction_idx].replace('_', ' ')
            session['confidence'] = round(float(np.max(outputs[0])), 2)
            
            # REDIRECT to GET request (PRG Pattern)
            return redirect(url_for('home'))
        except Exception as e:
            session['error'] = str(e)
            return redirect(url_for('home'))

    # GET Request Logic
    # Pull data from session and then clear it
    prediction = session.pop('prediction', None)
    confidence = session.pop('confidence', None)
    error = session.pop('error', None)

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
