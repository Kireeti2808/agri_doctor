import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import openai
import requests
from tensorflow.keras.applications import efficientnet, mobilenet

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Agri-Doctor Pro", layout="wide", page_icon="🌿")

# --- 2. CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    /* Global Rounding */
    .stButton>button {
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stTextInput>div>div>input {
        border-radius: 15px;
    }
    .css-1r6slb0 { /* Container Padding */
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Weather Widget Styling */
    .weather-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
    .weather-temp {
        font-size: 36px;
        font-weight: 800;
        color: #333;
        margin: 0;
    }
    .weather-desc {
        font-size: 14px;
        color: #555;
        text-transform: capitalize;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CONSTANTS ---
CORN_MODEL_ID = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
RICE_MODEL_ID = '1p2vZgq_FBigVnlhQPLQD4w2yjDn4zus3'
COTTON_DISEASE_ID = '14d3ZHEA8GnOliO164BA811tWnZ-EhPm0'
COTTON_WEED_ID = '1Sk2h23GtVLPHBJ700Ld4AEOU8ra9dRmH'

CORN_CLASSES = ['Maize_Blight', 'Maize_Common_Rust', 'Maize_Gray_Leaf_Spot', 'Maize_Healthy', 'Weed_Broadleaf', 'Weed_Grass']
RICE_CLASSES = ['Rice_Bacterial_Leaf_Blight', 'Rice_Brown_Spot', 'Rice_Healthy', 'Rice_Leaf_Blast', 'Rice_Leaf_scald', 'Rice_Sheath_Blight', 'Weed_Broadleaf', 'Weed_Grass']
COTTON_DISEASE_CLASSES = ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 'Herbicide Growth Damage', 'Leaf Redding', 'Leaf Variegation']
COTTON_WEED_CLASSES = ['Carpetweeds', 'Morningglory', 'PalmerAmaranth', 'Purslane', 'Waterhemp']

# --- 4. WEATHER VISUALS (PURE CSS) ---
def get_weather_visual(condition):
    condition = condition.lower()
    
    # CSS Shapes for Weather Icons
    sun_html = """
    <div style="width:60px; height:60px; background:#FFD700; border-radius:50%; box-shadow: 0 0 15px orange; margin:auto; animation: spin 10s linear infinite;"></div>
    """
    
    cloud_html = """
    <div style="width:70px; height:35px; background:#fff; border-radius:20px; margin:auto; position:relative; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>
    """
    
    rain_html = """
    <div style="width:60px; height:30px; background:#778899; border-radius:20px; margin:auto;"></div>
    <div style="display:flex; justify-content:center; gap:5px; margin-top:5px;">
        <div style="width:4px; height:10px; background:#00BFFF; border-radius:2px;"></div>
        <div style="width:4px; height:10px; background:#00BFFF; border-radius:2px;"></div>
        <div style="width:4px; height:10px; background:#00BFFF; border-radius:2px;"></div>
    </div>
    """
    
    storm_html = """
    <div style="width:60px; height:30px; background:#444; border-radius:20px; margin:auto;"></div>
    <div style="text-align:center; color:#FFD700; font-size:20px; line-height:10px;">⚡</div>
    """
    
    if "sun" in condition or "clear" in condition: return sun_html
    elif "rain" in condition or "shower" in condition or "drizzle" in condition: return rain_html
    elif "storm" in condition or "thunder" in condition: return storm_html
    else: return cloud_html # Default Cloudy

# --- 5. FUNCTIONS ---
@st.cache_resource
def load_model(model_key):
    if model_key == 'Maize': file_id, filename = CORN_MODEL_ID, 'corn_model.tflite'
    elif model_key == 'Rice': file_id, filename = RICE_MODEL_ID, 'rice_model.tflite'
    elif model_key == 'Cotton_Disease': file_id, filename = COTTON_DISEASE_ID, 'cotton_disease.tflite'
    elif model_key == 'Cotton_Weed': file_id, filename = COTTON_WEED_ID, 'cotton_weed.tflite'
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {model_key} Model..."):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)
    
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(image, interpreter, model_key):
    input_details = interpreter.get_input_details(); output_details = interpreter.get_output_details()
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    if 'Cotton' in model_key: img_array = mobilenet.preprocess_input(img_array)
    else: img_array = efficientnet.preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def get_weather(location):
    if not location: return None
    try:
        url = f"https://wttr.in/{location}?format=%t|%h|%p|%C"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text.split("|")
            temp = data[0].strip()
            # Force Celsius Visual
            if "F" in temp:
                 val = float(''.join(filter(str.isdigit, temp)))
                 temp = f"{int((val - 32) * 5/9)}°C"
            
            return {
                "temp": temp, "humidity": data[1].strip(), 
                "precip": data[2].strip(), "condition": data[3].strip(),
                "visual": get_weather_visual(data[3].strip())
            }
    except: return None
    return None

def analyze_quadrants(image, interpreter, classes, model_key):
    w, h = image.size; mid_w, mid_h = w // 2, h // 2
    quadrants = {
        "Top-Left": image.crop((0, 0, mid_w, mid_h)), "Top-Right": image.crop((mid_w, 0, w, mid_h)),
        "Bottom-Left": image.crop((0, mid_h, mid_w, h)), "Bottom-Right": image.crop((mid_w, mid_h, w, h))
    }
    results = {}
    for name, img_crop in quadrants.items():
        preds = predict_image(img_crop, interpreter, model_key)
        idx = np.argmax(preds); conf = np.max(preds) * 100
        results[name] = {"label": classes[idx], "conf": conf, "img": img_crop}
    return results

def get_smart_advice(diagnosis, weather, location, mode):
    try:
        client = openai.OpenAI(api_key=st.secrets["openai_key"])
        weather_txt = f"{weather['temp']}, {weather['condition']}" if weather else "Unknown"
        prompt = f"Act as Agronomist. Issue: {diagnosis} in {mode}. Weather: {weather_txt} in {location}. Give 3-step solution."
        return client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    except: return "AI Advice Unavailable"

# --- 6. MAIN UI ---
st.sidebar.title("Agri-Doctor 👨‍⚕️")
st.sidebar.markdown("---")

crop_choice = st.sidebar.radio("Select Crop", ["Maize (Corn)", "Rice (Paddy)", "Cotton"])

if crop_choice == "Cotton":
    st.header("☁️ Cotton Intelligence")
    analysis_type = st.sidebar.radio("Focus", ["Leaf Disease", "Weed Type"])
    model_key, current_classes = ("Cotton_Disease", COTTON_DISEASE_CLASSES) if analysis_type == "Leaf Disease" else ("Cotton_Weed", COTTON_WEED_CLASSES)
else:
    if crop_choice == "Maize (Corn)": st.header("🌽 Maize Health"); model_key, current_classes = "Maize", CORN_CLASSES
    else: st.header("🌾 Rice Health"); model_key, current_classes = "Rice", RICE_CLASSES

user_location = st.sidebar.text_input("Location", placeholder="e.g. Hyderabad")
enable_ai = st.sidebar.checkbox("Enable AI Advice", value=True)

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Rounded Image CSS hack
        st.markdown('<style>img {border-radius: 15px;}</style>', unsafe_allow_html=True)

    with col2:
        weather_data = get_weather(user_location)
        if weather_data:
            # --- WEATHER WIDGET (ROUNDED & STYLED) ---
            st.markdown(f"""
            <div class="weather-card">
                <div>
                    {weather_data['visual']}
                </div>
                <div>
                    <p class="weather-temp">{weather_data['temp']}</p>
                    <p class="weather-desc">{weather_data['condition']}</p>
                    <p style="font-size:12px; margin:0; color:#666;">Humidity: {weather_data['humidity']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button('🚀 Run Diagnosis'):
            with st.spinner('Analyzing...'):
                try:
                    interpreter = load_model(model_key)
                    quad_results = analyze_quadrants(image, interpreter, current_classes, model_key)
                    
                    st.write("### 🔍 Scan Results")
                    q1, q2 = st.columns(2)
                    detections = []
                    
                    for i, (name, res) in enumerate(quad_results.items()):
                        target_col = q1 if i % 2 == 0 else q2
                        with target_col:
                            st.image(res['img'], width=150)
                            
                            label = res['label']; conf = res['conf']
                            color = "#dc3545" # Red
                            if "Healthy" in label: color = "#28a745" # Green
                            if "Weed" in label or label in COTTON_WEED_CLASSES: color = "#fd7e14" # Orange
                            
                            # ROUNDED PROGRESS BAR
                            st.markdown(f"""
                                <div style="margin-bottom:15px; background:#f0f2f6; padding:10px; border-radius:12px;">
                                    <div style="font-weight:bold; color:{color}; font-size:13px; margin-bottom:5px;">{label}</div>
                                    <div style="width:100%; background:#e0e0e0; border-radius:10px; height:8px;">
                                        <div style="width:{conf}%; background:{color}; height:8px; border-radius:10px;"></div>
                                    </div>
                                    <div style="text-align:right; font-size:10px; color:#666;">{conf:.0f}%</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if conf > 50: detections.append(label)
                    
                    if enable_ai:
                        st.markdown("---")
                        final_issue = max(set(detections), key=detections.count) if detections else "Healthy/Unknown"
                        
                        # Custom Success Box
                        st.markdown(f"""
                        <div style="background-color:#d4edda; color:#155724; padding:15px; border-radius:15px; border:1px solid #c3e6cb; margin-bottom:10px;">
                            <strong>✅ Diagnosis Complete:</strong> {final_issue}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        advice = get_smart_advice(final_issue, weather_data, user_location, crop_choice)
                        st.info(advice)
                        
                except Exception as e: st.error(f"Error: {str(e)}")
