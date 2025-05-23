import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Cache the model to avoid reloading on every rerun
@st.cache_resource
def load_model():
    file_id = "YOUR_FILE_ID_HERE"  # replace with your actual file ID
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path = "best_weather_mlp_model.h5"

    if not os.path.exists(output_path):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    try:
        model = tf.keras.models.load_model(output_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

st.write("# VERDEJO FINAL EXAM")

file = st.file_uploader("Upload a weather photo", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)  # match your training image size
    image_data = image_data.convert('RGB')  # Ensure 3 channels

    # For Pillow < 9.1, use Image.LANCZOS
    resample_method = getattr(Image, 'Resampling', Image).LANCZOS  
    image = ImageOps.fit(image_data, size, resample_method)

    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
elif model is None:
    st.error("Model not loaded. Cannot make predictions.")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    
    prediction = import_and_predict(image, model)
    
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    predicted_label = class_names[np.argmax(prediction)]
    
    st.success(f"Weather Prediction: {predicted_label}")

    confidence = np.max(prediction) * 100
    st.info(f"Model confidence: {confidence:.2f}%")
