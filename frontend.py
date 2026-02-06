import streamlit as st
import requests
import base64
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Live Object Similarity Classification & Counting System")

# Camera input
camera_image = st.camera_input("Take a picture")

# File uploader
uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Combine images
images = []
if camera_image:
    images.append(Image.open(camera_image).convert('RGB'))
if uploaded_files:
    images.extend([Image.open(file).convert('RGB') for file in uploaded_files])

if uploaded_files:
    images = [Image.open(file).convert('RGB') for file in uploaded_files]

    # Convert images to base64
    images_data = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_data.append(f"data:image/jpeg;base64,{img_str}")

    # Send to backend
    response = requests.post('http://localhost:5000/process_images', json={'images': images_data})
    if response.status_code == 200:
        result = response.json()

        st.subheader("Results")
        st.write(f"Total objects: {result['total_objects']}")
        st.write(f"Number of clusters: {result['clusters']}")

        df = pd.DataFrame(result['counts'])
        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.bar([str(c['cluster']) for c in result['counts']], [c['count'] for c in result['counts']])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.set_title('Objects per Cluster')
        st.pyplot(fig)

        # Display images grouped by cluster
        labels = result['labels']
        unique_labels = set(labels)
        for label in unique_labels:
            st.write(f"Cluster {label}:")
            cluster_images = [img for img, l in zip(images, labels) if l == label]
            cols = st.columns(len(cluster_images))
            for i, img in enumerate(cluster_images):
                cols[i].image(img, use_column_width=True)
    else:
        st.error("Error processing images")
