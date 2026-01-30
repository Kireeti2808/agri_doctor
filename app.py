import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

MODEL_FILE_ID = '1NzmXgv3nDe0xorHoxhWF06cYLd21UMM4'
MODEL_FILENAME = 'corn_model.h5'

CLASS_NAMES = [
    'Maize_Blight',
    'Maize_Common_Rust',
    'Maize_Gray_Leaf_Spot',
    'Maize_Healthy',
    'Weed_Broadleaf',
    'Weed_Grass'
]

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILENAME):
        file_size = os.path.getsize(MODEL_FILENAME)
        if file_size < 1000000:
            os.remove(MODEL_FILENAME)

    if not os.path.exists(MODEL_FILENAME):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_FILENAME, quiet=False, fuzzy=True)

    model = tf.keras.models.load_model(MODEL_FILENAME, compile=False, safe_mode=False)
    return model

def predict_image(image, model):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array)
    return predictions

st.set_page_config(page_title="Maize Doctor")
st.title("Maize and Weed Doctor")
st.write("Upload a photo of a maize leaf or weed to detect diseases.")

model = None

try:
    with st.spinner("Loading Model..."):
        model = load_model()
    st.success("System Ready")
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Analyze Plant"):
        with st.spinner("Analyzing..."):
            preds = predict_image(image, model)
            class_idx = np.argmax(preds)
            confidence = np.max(preds)
            predicted_label = CLASS_NAMES[class_idx]

            st.write("---")
            st.subheader(f"Result: {predicted_label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")

            if "Healthy" in predicted_label:
                st.success("Plant is healthy.")
            elif "Weed" in predicted_label:
                st.warning("Weed detected.")
            else:
                st.error("Disease detected.")
