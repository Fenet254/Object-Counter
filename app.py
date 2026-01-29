import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import sklearn.cluster as cluster
from sklearn.metrics.pairwise import cosine_similarity
import io

# Load pre-trained ResNet50 model
model = resnet50(pretrained=True)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    """Extract features from an image using ResNet50."""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()

def cluster_objects(features, n_clusters=None):
    """Cluster objects based on features."""
    if n_clusters is None:
        # Use DBSCAN for automatic clustering
        db = cluster.DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        labels = db.fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
    return labels, n_clusters

def main():
    st.title("Object Counter")
    st.write("Upload images or use camera to count similar objects.")

    option = st.selectbox("Choose input method:", ["Upload Images", "Camera"])

    images = []
    features_list = []

    if option == "Upload Images":
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                images.append(image)
                features = extract_features(image)
                features_list.append(features)
    elif option == "Camera":
        st.write("Camera functionality not implemented yet. Please upload images.")

    if images:
        features_array = np.array(features_list)
        labels, n_clusters = cluster_objects(features_array)

        # Count objects in each cluster
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        total_count = len(images)

        st.write(f"Total objects: {total_count}")
        st.write(f"Number of clusters: {n_clusters}")
        st.write("Cluster counts:")
        st.dataframe(cluster_counts)

        # Display images grouped by cluster
        for cluster_id in range(n_clusters):
            st.write(f"Cluster {cluster_id}: {cluster_counts[cluster_id]} objects")
            cluster_images = [img for img, label in zip(images, labels) if label == cluster_id]
            cols = st.columns(min(len(cluster_images), 5))
            for i, img in enumerate(cluster_images[:5]):
                cols[i].image(img, caption=f"Object {i+1}", use_column_width=True)

if __name__ == "__main__":
    main()
