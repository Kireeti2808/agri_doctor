import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import gdown
import os
import openai

CORN_MODEL_ID = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
RICE_MODEL_ID = '1p2vZgq_FBigVnlhQPLQD4w2yjDn4zus3'

CORN_CLASSES = [
    'Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 
    'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass'
]

RICE_CLASSES = [
    'Rice_Bacterial_Leaf_Blight', 'Rice_Brown_Spot', 'Rice_Healthy_Rice_Leaf',
    'Rice_Leaf_Blast', 'Rice_Leaf_scald', 'Rice_Sheath_Blight',
    'Weed_Broadleaf', 'Weed_Grass'
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

def quadrant_analysis(image, interpreter, predicted_index):
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    height, width = 224, 224
    mid_h, mid_w = height // 2, width // 2
    
    quadrants = {
        "Top-Left": (0, 0, mid_h, mid_w),
        "Top-Right": (0, mid_w, mid_h, width),
        "Bottom-Left": (mid_h, 0, height, mid_w),
        "Bottom-Right": (mid_h, mid_w, height, width)
    }
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    orig_input = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(img_array.copy()), axis=0)
    interpreter.set_tensor(input_details[0]['index'], orig_input)
    interpreter.invoke()
    orig_score = interpreter.get_tensor(output_details[0]['index'])[0][predicted_index]
    
    drops = {}
    
    for name, (y1, x1, y2, x2) in quadrants.items():
        masked_img = img_array.copy()
        masked_img[y1:y2, x1:x2, :] = 0 
        
        inp = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(masked_img), axis=0)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        new_score = interpreter.get_tensor(output_details[0]['index'])[0][predicted_index]
        
        drops[name] = max(0, orig_score - new_score)
        
    return drops

def get_gpt_advice(disease_name, location):
    if "Healthy" in disease_name:
        return "The plant is healthy. Keep up the good work with regular watering and monitoring."
    
    try:
        api_key = st.secrets["openai_key"]
    except:
        return "OpenAI API Key missing in Secrets."

    client = openai.OpenAI(api_key=api_key)
    
    loc_text = f"in {location}" if location else ""
    
    prompt = f"""
    You are an expert Agronomist. A farmer {loc_text} has detected '{disease_name}' in their crop.
    Provide a concise response considering the location/climate if relevant:
    1. Cause (1 sentence)
    2. Immediate Cure (Chemical or Organic suitable for this region)
    3. Prevention for next season.
    Keep it simple and actionable.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI Doctor: {e}"

st.set_page_config(page_title="Agri-Doctor Pro", layout="wide")

st.sidebar.title("Agri-Doctor")
st.sidebar.subheader("Settings")
crop_choice = st.sidebar.radio("Select Crop", ["Maize (Corn)", "Rice (Paddy)"])
user_location = st.sidebar.text_input("Enter Your Location", placeholder="e.g., Hyderabad, India")
enable_ai = st.sidebar.checkbox("Enable AI Advice (OpenAI)", value=True)
enable_xai = st.sidebar.checkbox("Enable Visual Analysis", value=False)

if crop_choice == "Maize (Corn)":
    st.title("Maize Disease Detection")
    current_classes = CORN_CLASSES
    model_key = 'Maize'
else:
    st.title("Rice Disease Detection")
    current_classes = RICE_CLASSES
    model_key = 'Rice'

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)
    
    with col2:
        if st.button('Run Diagnosis'):
            with st.spinner('Analyzing patterns...'):
                try:
                    interpreter = load_model(model_key)
                    predictions = predict_image(image, interpreter)
                    idx = np.argmax(predictions)
                    confidence = np.max(predictions) * 100
                    result_class = current_classes[idx]
                    
                    st.success(f"Diagnosis: {result_class}")
                    st.metric("Confidence", f"{confidence:.2f}%")
                    
                    if enable_xai:
                        st.write("Visual Quadrant Analysis")
                        with st.spinner("Checking quadrants..."):
                            drops = quadrant_analysis(image, interpreter, idx)
                            important_quadrant = max(drops, key=drops.get)
                            
                            st.bar_chart(drops)
                            st.info(f"Most critical area: {important_quadrant}")

                    if enable_ai:
                        st.write("AI Doctor Prescription")
                        with st.spinner("Consulting GPT..."):
                            advice = get_gpt_advice(result_class, user_location)
                            st.info(advice)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
