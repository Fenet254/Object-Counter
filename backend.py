from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)

# Load pre-trained ResNet50 model
model = resnet50(pretrained=True)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze().numpy()

def cluster_objects(features, eps=0.5, min_samples=1):
    similarity = cosine_similarity(features)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(1 - similarity)
    return labels

@app.route('/process_images', methods=['POST'])
def process_images():
    data = request.get_json()
    images_data = data['images']

    images = []
    features = []
    for img_data in images_data:
        img_str = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        images.append(img)
        feat = extract_features(img)
        features.append(feat)

    features = np.array(features)
    labels = cluster_objects(features)

    unique_labels, counts = np.unique(labels, return_counts=True)
    total_objects = len(images)
    clusters = len(unique_labels)

    counts_list = [{'cluster': int(label), 'count': int(count)} for label, count in zip(unique_labels, counts)]

    return jsonify({
        'total_objects': total_objects,
        'clusters': clusters,
        'counts': counts_list,
        'labels': labels.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
