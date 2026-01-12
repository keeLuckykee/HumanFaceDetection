import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# 1. Initialize session state
if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = None

# --- a) Retrieve Model from Google Drive Folder ---
@st.cache_resource
def load_emotion_model():
    model_name = "full_emotion_model.keras"
    # The folder link you provided
    folder_url = "https://drive.google.com/drive/folders/1gd8pl9pPY0XZe4IJV8ho-WNHcnrsWFOf"
    
    if not os.path.exists(model_name):
        with st.spinner('Downloading model from Google Drive...'):
            # gdown will download the contents of the folder
            gdown.download_folder(folder_url, quiet=False, use_cookies=False)
            
    # Load the specific .keras file
    return tf.keras.models.load_model(model_name, compile=False)

model = load_emotion_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("Human Emotion Detection Web App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image for display
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- b) Grayscale to RGB Conversion ---
    # Model expects (48, 48, 3). convert('RGB') handles gray images automatically
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and normalize
    image = image.resize((48, 48))
    img_array = np.array(image) / 255.0
    img_batch = np.expand_dims(img_array, axis=0) # Shape: (1, 48, 48, 3)

    # --- c) Prediction Logic ---
    if st.button("Predict Emotion"):
        with st.spinner('Analyzing...'):
            predictions = model.predict(img_batch)
            max_index = np.argmax(predictions[0])
            
            # Store in session state
            st.session_state.prediction_label = emotion_labels[max_index]
            st.session_state.confidence_score = predictions[0][max_index] * 100

# Display results consistently
if st.session_state.prediction_label:
    st.success(f"Result: {st.session_state.prediction_label.upper()}")
    st.info(f"Confidence: {st.session_state.confidence_score:.2f}%")
