import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os

# -------------------------------
# Load model (cached, safe)
# -------------------------------
@st.cache_resource
def load_emotion_model():
    # Using the path provided in your original file
    model_path = r"C:\Users\cyber\HumanFaceDetectionProject\full_emotion_model.keras"
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_emotion_model()
st.success("Emotion model loaded successfully!")

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(uploaded_file):
    # Load image using PIL
    img = Image.open(uploaded_file)
    # Convert to grayscale first (if it's already gray, this ensures consistency)
    img = img.convert('L') 
    # Resize to 48x48 as required by your model
    img = img.resize((48, 48))
    img_array = np.array(img)
    
    # FIX: Convert Grayscale (1 channel) to RGB (3 channels) 
    # This solves the "expected axis -1 to have value 3" error
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Rescale pixels (usually 0-1 or -1 to 1 depending on your training)
    img_rgb = img_rgb / 255.0
    
    # Add batch dimension: (1, 48, 48, 3)
    img_batch = np.expand_dims(img_rgb, axis=0)
    return img_batch

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Facial Emotion Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and Predict
    processed_img = preprocess_image(uploaded_file)
    prediction = model.predict(processed_img)
    
    # Mapping classes (Update these based on your specific notebook labels)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    result = emotion_labels[np.argmax(prediction)]
    
    st.write(f"### Predicted Emotion: {result}")
