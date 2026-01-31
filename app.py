import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

CORN_MODEL_ID = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk' 
RICE_MODEL_ID = '1p2vZgq_FBigVnlhQPLQD4w2yjDn4zus3'

CORN_CLASSES = [
    'Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 
    'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass'
]

RICE_CLASSES = [
    'Rice_Bacterial_Leaf_Blight',
    'Rice_Brown_Spot',
    'Rice_Healthy_Rice_Leaf',
    'Rice_Leaf_Blast',
    'Rice_Leaf_scald',
    'Rice_Sheath_Blight',
    'Weed_Broadleaf',
    'Weed_Grass'
]

@st.cache_resource
def load_model(crop_type):
    if crop_type == 'Maize':
        file_id = CORN_MODEL_ID
        filename = 'corn_model.tflite'
    else:
        file_id = RICE_MODEL_ID
        filename = 'rice_model.tflite'
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {crop_type} Model..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)
    
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data[0]

st.set_page_config(page_title="Agri-Doctor", layout="wide")

st.sidebar.title("Agri-Doctor")
st.sidebar.subheader("Select Crop")
crop_choice = st.sidebar.radio("", ["Maize (Corn)", "Rice (Paddy)"])

if crop_choice == "Maize (Corn)":
    st.title("Maize Disease Detection")
    st.write("Upload a maize leaf to detect Blight, Rust, or Leaf Spot.")
    current_classes = CORN_CLASSES
    model_key = 'Maize'

else:
    st.title("Rice & Weed Doctor")
    st.write("Upload a rice leaf to detect Diseases or Weeds.")
    current_classes = RICE_CLASSES
    model_key = 'Rice'

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    with col2:
        st.write("### Analysis Result")
        if st.button('Analyze Plant'):
            with st.spinner(f'Processing...'):
                try:
                    interpreter = load_model(model_key)
                    
                    predictions = predict_image(image, interpreter)
                    
                    idx = np.argmax(predictions)
                    confidence = np.max(predictions) * 100
                    result_class = current_classes[idx]
                    
                    if confidence > 50:
                        st.success(f"Diagnosis: {result_class}")
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                        
                        if "Healthy" in result_class:
                            st.write("Plant is healthy!")
                        elif "Weed" in result_class:
                            st.warning("Weed detected! Recommended: Remove manually or apply herbicide.")
                        else:
                            st.warning("Disease detected. Recommended: Isolate plant and check fungicide options.")
                    else:
                        st.error("Uncertain prediction. The leaf might be unclear or not belong to this crop.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.write("Tip: Check if the Google Drive ID is valid and public.")



