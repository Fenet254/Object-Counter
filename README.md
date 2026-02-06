# Object-Counter

A real-time AI-powered system for object similarity classification and counting using webcam or uploaded images.

## Features

- Feature extraction using ResNet50
- Similarity-based clustering with DBSCAN
- Real-time counting of objects per group
- Interactive Streamlit UI

## Installation

1. Install Python 3.11 or later.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the app:
```
streamlit run app.py
```

Upload images to classify and count similar objects.

## Requirements

- Python 3.11+
- PyTorch
- OpenCV
- Scikit-learn
- Streamlit
