# Project Mandates: CIFAR-100 ResNet Inference API

This document outlines the requirements and current status for the CIFAR-100 ResNet project. These instructions take precedence over general workflows.

## 1. Project Overview
- **Dataset:** CIFAR-100 (32x32 images, 100 classes).
- **Architecture:** ResNet-18 (modified for 32x32 input).
- **Export Format:** ONNX.
- **Inference Engine:** ONNX Runtime (`onnxruntime`).
- **Web Framework:** Flask.

## 2. Requirements & Constraints
- **Preprocessing Pipeline:**
  1. Resize to exactly 32x32 pixels.
  2. Convert to NumPy array.
  3. Normalize using CIFAR-100 mean (0.5071, 0.4867, 0.4408) and std (0.2675, 0.2565, 0.2761).
  4. Transpose dimensions to (Channels, Height, Width).
- **API Endpoints:**
  - `POST /predict`: Must accept an image file (multipart/form-data) and return a JSON response containing `class_id`, `class_name`, and `confidence`.
  - `GET /`: Should serve a simple UI (existing `index.html`).

## 3. Current Status
- [x] Training script (`train.py`) implemented with ONNX export.
- [x] Initial `app.py` created (needs `/predict` endpoint).
- [x] ONNX model (`model.onnx`) generated.
- [ ] `/predict` JSON endpoint in `app.py`.
- [ ] Docker image for the inference API.

## 4. Development Strategy
1. **API Update:** Modify `app.py` to include the mandatory `/predict` JSON endpoint.
2. **Local Validation:** Verify the API with a sample image using a script or `curl`.
3. **Containerization:** Create a `Dockerfile` for the inference API.
4. **Final Verification:** Build and run the Docker container to ensure full functionality.
