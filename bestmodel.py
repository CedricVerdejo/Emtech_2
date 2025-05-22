import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Caches the model to prevent reloading on every rerun
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_weather_mlp_model.h5')
    return model

model = load_model()

st.write("""
# VERDEJO FINAL EXAM
""")

file = st.file_uploader("Upload a weather photo", type=["jpg", "png"])

# Prediction function
def import_and_predict(image_data, model):
    size = (64, 64)  # match with your training image size
    image_data = image_data.convert('RGB')  # Ensure 3 channels
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)
    prediction = model.predict(img_reshape)
    return prediction

# Main app logic
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    
    prediction = import_and_predict(image, model)
    
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_label = class_names[np.argmax(prediction)]
    
    st.success(f"Weather Prediction: {predicted_label}")

    confidence = np.max(prediction) * 100
    st.info(f"Model confidence: {confidence:.2f}%")
