import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os
import gdown
import openai

MODEL_FILE_ID = '1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk'
MODEL_FILENAME = 'corn_model.tflite'

CLASS_NAMES = {
    0: 'Maize_Blight',
    1: 'Maize_Common_Rust',
    2: 'Maize_Gray_Leaf_Spot',
    3: 'Maize_Healthy',
    4: 'Weed_Broadleaf',
    5: 'Weed_Grass'
}

st.set_page_config(page_title="Agri-Smart Advisor", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .disease-gradient {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    }
    .weed-gradient {
        background: linear-gradient(135deg, #fce38a 0%, #f38181 100%);
        color: #4a4a4a;
    }
    .healthy-gradient {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #ddd;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_FILENAME):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_FILENAME, quiet=False, fuzzy=True)

    interpreter = tf.lite.Interpreter(model_path=MODEL_FILENAME)
    interpreter.allocate_tensors()
    return interpreter

def tflite_predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def get_real_weather(city_name):
    try:
        url = f"https://wttr.in/{city_name}?format=j1"
        response = requests.get(url)
        data = response.json()
        current = data['current_condition'][0]
        return {
            "temperature": current['temp_C'],
            "humidity": current['humidity'],
            "condition": current['weatherDesc'][0]['value'],
            "city": city_name
        }
    except:
        return {"temperature": "30", "humidity": "45", "condition": "Sunny", "city": city_name}

def get_openai_advice(vision_results, weather):
    if "OPENAI_API_KEY" not in st.secrets:
        return "OpenAI API Key missing. Please check your Secrets."

    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    problems = ", ".join(list(vision_results))
    
    prompt = f"""
    You are an expert agronomist.
    Situation:
    - Crops Issues: {problems}
    - Weather: {weather['condition']}, {weather['temperature']}C
    
    Task:
    Provide a formatted advice section:
    1. **Immediate Action**: What to do right now.
    2. **Chemical Solution**: Specific fungicide/herbicide names.
    3. **Organic Solution**: Home-made or natural remedies.
    Keep it concise.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {e}"

st.title("Agri-Smart Advisor")

st.sidebar.header("Settings")
city = st.sidebar.text_input("Farm Location (City)", value="Vellore")

try:
    interpreter = load_model_from_drive()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "png", "jpeg"])

if uploaded_file and city:
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(original_img, caption="Uploaded Field Image", use_column_width=True)
            
        with col2:
            weather = get_real_weather(city)
            st.subheader(f"Weather in {weather['city']}")
            w_c1, w_c2 = st.columns(2)
            w_c1.metric("Temperature", f"{weather['temperature']}C")
            w_c2.metric("Humidity", f"{weather['humidity']}%")
            st.info(f"Condition: {weather['condition']}")

        st.divider()

        with st.spinner("AI is scanning the crops..."):
            width, height = original_img.size
            crops = {
                "Top-Left": original_img.crop((0, 0, width//2, height//2)),
                "Top-Right": original_img.crop((width//2, 0, width, height//2)),
                "Bottom-Left": original_img.crop((0, height//2, width//2, height)),
                "Bottom-Right": original_img.crop((width//2, height//2, width, height))
            }
            
            found_problems = set()
            display_cards = []

            for pos, crop_img in crops.items():
                preds = tflite_predict(interpreter, crop_img)
                pred_index = np.argmax(preds)
                confidence = np.max(preds) * 100
                label = CLASS_NAMES[pred_index]
                
                if confidence > 55:
                    found_problems.add(label)
                    display_cards.append({
                        "pos": pos,
                        "label": label,
                        "conf": confidence
                    })

        st.markdown('<div class="section-title">1. Detection Results</div>', unsafe_allow_html=True)
        
        if not found_problems:
             st.markdown(f"""
                <div class="result-card healthy-gradient">
                    <h3>Healthy Crop</h3>
                    <p>No diseases or weeds detected with high confidence.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            cols = st.columns(len(display_cards)) if len(display_cards) <= 2 else st.columns(2)
            
            for i, card in enumerate(display_cards):
                if "Healthy" in card['label']:
                    css_class = "healthy-gradient"
                elif "Weed" in card['label']:
                    css_class = "weed-gradient"
                else:
                    css_class = "disease-gradient"

                html_card = f"""
                <div class="result-card {css_class}">
                    <h4>{card['label']}</h4>
                    <p><b>Location:</b> {card['pos']}</p>
                    <p><b>Confidence:</b> {card['conf']:.1f}%</p>
                </div>
                """
                with cols[i % 2]:
                    st.markdown(html_card, unsafe_allow_html=True)
                    st.image(crops[card['pos']], width=150)

        if found_problems:
            st.divider()
            st.markdown('<div class="section-title">2. AI Treatment Plan</div>', unsafe_allow_html=True)
            
            with st.spinner("Consulting AI Agronomist for treatment..."):
                advice = get_openai_advice(found_problems, weather)
                st.markdown(advice)
            
            st.warning("Disclaimer: Always consult a local agricultural expert before applying chemicals.")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
