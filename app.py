import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os
import gdown  # Optimized for Google Drive downloads

# -------------------------------
# Model Downloading & Loading
# -------------------------------
@st.cache_resource
def load_emotion_model():
    model_path = "full_emotion_model.keras"
    
    # If model doesn't exist on the server, download it
    if not os.path.exists(model_path):
        with st.spinner('Downloading model from cloud storage... this may take a moment.'):
            # Replace with your actual direct download link
            # For Google Drive: use 'https://drive.google.com/uc?id=FILE_ID'
            url = "https://drive.google.com/drive/folders/1gd8pl9pPY0XZe4IJV8ho-WNHcnrsWFOf"
            gdown.download(url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_emotion_model()
st.success("Emotion model ready!")


# -------------------------------
# Image Preprocessing (48x48x3)
# -------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((48, 48))
    img_array = np.array(img)
    
    # Check if image is grayscale; if so, convert to RGB
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4: # Handle RGBA (transparent) images
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# -------------------------------
# UI Logic
# -------------------------------
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Input Image")
    processed = preprocess_image(uploaded_file)
    
    # The model expects (batch, 48, 48, 3)
    prediction = model.predict(processed)
    
    # Emotion labels from your notebook
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    st.write(f"## Prediction: {labels[np.argmax(prediction)]}")

