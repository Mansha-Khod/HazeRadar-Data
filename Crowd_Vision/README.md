
# CrowdVision

CrowdVision is the image-based haze detection module of AirCast.

It uses a deep learning model to classify uploaded images as:
- Clear
- Hazy

The module is built using EfficientNet-B0 with transfer learning in PyTorch.

## Features

- Image upload API
- Real-time haze prediction
- Confidence score generation
- Visibility score analysis
- Atmospheric haze detection

## Tech Stack

- Python
- Flask
- PyTorch
- OpenCV
- PIL

## Model

Architecture:
- EfficientNet-B0

Classes:
- Clear
- Hazy
