import streamlit as st
from tensorflow.keras.models import load_model
import os


# -------------------------------
# Load model (cached, safe)
# -------------------------------
@st.cache_resource
def load_emotion_model():
    # Using a raw string (r"") to handle Windows backslashes correctly
    model_path = r"C:\Users\cyber\HumanFaceDetectionProject\full_emotion_model.keras"
    
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )
    return model

#@st.cache_resource
#def load_emotion_model():
#    model_path = "full_emotion_model.keras"
#    if not os.path.exists(model_path):
#       st.error("Model file not found!")
#       st.stop()
#    return load_model(model_path)

#model = load_emotion_model()

st.success("Emotion model loaded successfully!")
