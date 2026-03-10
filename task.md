Dataset - CIFAR-100

What architecture to use? 
VGGNet
Inception (v1-v4)
Inception-ResNet-v2
ResNet
Xception

1. Train the model
2. Convert model to Tensorflow Lite or ONNX
3. Setup the inference API server using any web-framework (Flask, FastAPI, etc.). The inference API must contain /predict endpoint, which accepts an image and returns model prediction;
4. Create Docker image for the inference API.
