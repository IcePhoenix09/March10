Dataset - CIFAR-100

Architecture - ResNet


1. Train the model
2. Convert model to Tensorflow Lite or ONNX
3. Setup the inference API server using any web-framework (Flask, FastAPI, etc.). The inference API must contain /predict endpoint, which accepts an image and returns model prediction;
4. Create Docker image for the inference API.

Install:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


the API will use ONNX Runtime (onnxruntime)

Preprocessing Pipeline: Since the model was trained on 32x32 images with specific normalization, the API must:
       1. Resize the incoming image to exactly 32x32 pixels.
       2. Convert it to a NumPy array.
       3. Normalize it using the CIFAR-100 mean (0.5071, 0.4867, 0.4408) and standard deviation (0.2675, 0.2565, 0.2761).
       4. Transpose the dimensions from (Height, Width, Channels) to (Channels, Height, Width) to match the model's input format.


The /predict Endpoint Logic
   * Input: A POST request containing an image file (multipart/form-data).
   * Validation: The script checks if an image was actually uploaded.
   * Inference:
       1. The image passes through the preprocessing pipeline.
       2. The ONNX model runs a "forward pass" on the image data.
       3. The model outputs a vector of 100 probabilities (one for each CIFAR-100 class).
   * Post-processing: The script finds the index with the highest probability (argmax).
   * Output: A JSON response containing the predicted class ID (and ideally the class name, like "beaver" or "maple_tree").
