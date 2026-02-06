import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained ResNet50 model
model = resnet50(pretrained=True)
model.eval()
# Remove the last layer to get features
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    """Extract features from an image using ResNet50."""
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze().numpy()

def cluster_objects(features, eps=0.5, min_samples=1):
    """Cluster objects based on features using DBSCAN."""
    # Compute cosine similarity
    similarity = cosine_similarity(features)
    # Use DBSCAN on the similarity matrix
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(1 - similarity)  # 1 - similarity for distance
    return labels

def main():
    st.title("Live Object Similarity Classification & Counting System")
    st.write("Upload images of objects to classify and count them based on visual similarity.")

    # File uploader for multiple images
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images = []
        features = []
        for file in uploaded_files:
            image = Image.open(file).convert('RGB')
            images.append(image)
            feat = extract_features(image)
            features.append(feat)

        features = np.array(features)

        # Clustering
        labels = cluster_objects(features)

        # Count objects per cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_objects = len(images)

        # Display results
        st.subheader("Clustering Results")
        st.write(f"Total objects: {total_objects}")
        st.write(f"Number of clusters: {len(unique_labels)}")

        # Table of counts
        df = pd.DataFrame({'Cluster': unique_labels, 'Count': counts})
        st.dataframe(df)

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(unique_labels.astype(str), counts)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.set_title('Objects per Cluster')
        st.pyplot(fig)

        # Display images grouped by cluster
        st.subheader("Grouped Images")
        for label in unique_labels:
            st.write(f"Cluster {label}:")
            cluster_images = [img for img, l in zip(images, labels) if l == label]
            cols = st.columns(len(cluster_images))
            for i, img in enumerate(cluster_images):
                cols[i].image(img, use_column_width=True)

if __name__ == "__main__":
    main()
