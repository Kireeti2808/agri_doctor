import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import openai
import requests
from tensorflow.keras.applications import efficientnet, mobilenet

st.set_page_config(page_title="Agri-Doctor Pro", layout="wide")

CORN_MODEL_ID = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
RICE_MODEL_ID = '1p2vZgq_FBigVnlhQPLQD4w2yjDn4zus3'
COTTON_DISEASE_ID = '14d3ZHEA8GnOliO164BA811tWnZ-EhPm0'
COTTON_WEED_ID = '1Sk2h23GtVLPHBJ700Ld4AEOU8ra9dRmH'

CORN_CLASSES = [
    'Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 
    'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass'
]

RICE_CLASSES = [
    'Rice_Bacterial_Leaf_Blight', 'Rice_Brown_Spot', 'Rice_Healthy', 
    'Rice_Leaf_Blast', 'Rice_Leaf_scald', 'Rice_Sheath_Blight', 
    'Weed_Broadleaf', 'Weed_Grass'
]

COTTON_DISEASE_CLASSES = [
    'Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 
    'Herbicide Growth Damage', 'Leaf Redding', 'Leaf Variegation'
]

COTTON_WEED_CLASSES = [
    'Carpetweeds', 'Morningglory', 'PalmerAmaranth', 'Purslane', 'Waterhemp'
]

@st.cache_resource
def load_model(model_key):
    if model_key == 'Maize':
        file_id, filename = CORN_MODEL_ID, 'corn_model.tflite'
    elif model_key == 'Rice':
        file_id, filename = RICE_MODEL_ID, 'rice_model.tflite'
    elif model_key == 'Cotton_Disease':
        file_id, filename = COTTON_DISEASE_ID, 'cotton_disease.tflite'
    elif model_key == 'Cotton_Weed':
        file_id, filename = COTTON_WEED_ID, 'cotton_weed.tflite'
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {model_key} Model..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)
    
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(image, interpreter, model_key):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    if 'Cotton' in model_key:
        img_array = mobilenet.preprocess_input(img_array)
    else:
        img_array = efficientnet.preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def get_weather(location):
    if not location: return None
    try:
        # UPDATED: Added '&M' to force Metric units (Celsius)
        url = f"https://wttr.in/{location}?format=%t|%h|%p|%C&M"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text.split("|")
            temp = data[0].strip()
            
            # Backup: If API still returns Fahrenheit (e.g. "86°F"), convert it manually
            if "F" in temp:
                 # Extract number and convert to Celsius
                 val = float(''.join(filter(str.isdigit, temp)))
                 celsius = int((val - 32) * 5/9)
                 temp = f"{celsius}°C"
            elif "C" not in temp:
                 # If it's just a number, assume it is now Celsius and add symbol
                 temp = f"{temp}"

            return {
                "temp": temp, 
                "humidity": data[1].strip(), 
                "precip": data[2].strip(), 
                "condition": data[3].strip()
            }
    except: return None
    return None

def analyze_quadrants(image, interpreter, classes, model_key):
    w, h = image.size
    mid_w, mid_h = w // 2, h // 2
    quadrants = {
        "Top-Left": image.crop((0, 0, mid_w, mid_h)), 
        "Top-Right": image.crop((mid_w, 0, w, mid_h)),
        "Bottom-Left": image.crop((0, mid_h, mid_w, h)), 
        "Bottom-Right": image.crop((mid_w, mid_h, w, h))
    }
    results = {}
    for name, img_crop in quadrants.items():
        preds = predict_image(img_crop, interpreter, model_key)
        idx = np.argmax(preds)
        conf = np.max(preds) * 100
        results[name] = {"label": classes[idx], "conf": conf, "img": img_crop}
    return results

def get_smart_advice(diagnosis, weather, location, mode):
    try:
        api_key = st.secrets["openai_key"]
        client = openai.OpenAI(api_key=api_key)
        
        weather_txt = f"{weather['temp']}, Hum: {weather['humidity']}" if weather else "Unknown"
        prompt = f"""
        Act as an Expert Agronomist. 
        Context: User detected '{diagnosis}' in {mode} Mode.
        Weather: {weather_txt} in {location}.
        
        Task: Provide a concise 3-step solution (Chemical & Organic).
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except: return "AI Advice Unavailable"

st.sidebar.title("Agri-Doctor")
st.sidebar.markdown("---")

crop_choice = st.sidebar.radio("Select Crop", ["Maize (Corn)", "Rice (Paddy)", "Cotton"])

if crop_choice == "Cotton":
    st.header("Cotton Intelligence System")
    analysis_type = st.sidebar.radio("Analysis Focus", ["Leaf Disease", "Weed Type"])
    
    if analysis_type == "Leaf Disease":
        model_key = "Cotton_Disease"
        current_classes = COTTON_DISEASE_CLASSES
    else:
        model_key = "Cotton_Weed"
        current_classes = COTTON_WEED_CLASSES
else:
    if crop_choice == "Maize (Corn)":
        st.header("Maize Disease Detection")
        model_key = "Maize"
        current_classes = CORN_CLASSES
    else:
        st.header("Rice Disease Detection")
        model_key = "Rice"
        current_classes = RICE_CLASSES

user_location = st.sidebar.text_input("Location", placeholder="e.g. Hyderabad")
enable_ai = st.sidebar.checkbox("Enable AI Advice", value=True)

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        weather_data = get_weather(user_location)
        if weather_data:
            st.metric("Current Weather", f"{weather_data['temp']} | {weather_data['condition']}")
        
        if st.button('Run Analysis'):
            with st.spinner('Analyzing...'):
                try:
                    interpreter = load_model(model_key)
                    quad_results = analyze_quadrants(image, interpreter, current_classes, model_key)
                    
                    st.write("### Quadrant Analysis")
                    q1, q2 = st.columns(2)
                    detections = []
                    
                    for i, (name, res) in enumerate(quad_results.items()):
                        target_col = q1 if i % 2 == 0 else q2
                        with target_col:
                            st.image(res['img'], width=150)
                            
                            label = res['label']
                            conf = res['conf']
                            color = "#dc3545"
                            if "Healthy" in label: color = "#28a745"
                            if "Weed" in label or label in COTTON_WEED_CLASSES: color = "#fd7e14"
                            
                            st.markdown(f"<span style='color:{color}; font-weight:bold'>{label} ({conf:.0f}%)</span>", unsafe_allow_html=True)
                            
                            if conf > 50: detections.append(label)
                    
                    if enable_ai:
                        st.markdown("---")
                        final_issue = max(set(detections), key=detections.count) if detections else "Healthy/Unknown"
                        
                        st.success(f"Primary Diagnosis: {final_issue}")
                        
                        advice = get_smart_advice(final_issue, weather_data, user_location, crop_choice)
                        st.info(advice)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
