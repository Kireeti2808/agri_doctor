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

def generate_heatmap(image, interpreter, predicted_index):
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    step = 28
    box = 40
    width, height = 224, 224
    heatmap = np.zeros((width, height))
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    orig_input = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(img_array.copy()), axis=0)
    interpreter.set_tensor(input_details[0]['index'], orig_input)
    interpreter.invoke()
    orig_score = interpreter.get_tensor(output_details[0]['index'])[0][predicted_index]
    
    for x in range(0, width, step):
        for y in range(0, height, step):
            masked_img = img_array.copy()
            masked_img[y:y+box, x:x+box, :] = 0
            
            inp = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(masked_img), axis=0)
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            new_score = interpreter.get_tensor(output_details[0]['index'])[0][predicted_index]
            
            heatmap[y:y+box, x:x+box] = max(0, orig_score - new_score)
            
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    return heatmap

def get_gpt_advice(disease_name):
    if "Healthy" in disease_name:
        return "The plant is healthy. Keep up the good work with regular watering and monitoring."
    
    if "openai_key" in st.secrets:
        api_key = st.secrets["openai_key"]
    else:
        return "OpenAI API Key missing in Secrets."

    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    You are an expert Agronomist. A farmer has detected '{disease_name}' in their crop.
    Provide a concise response with:
    1. Cause (1 sentence)
    2. Immediate Cure (Chemical or Organic)
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
                        st.write("Visual Analysis")
                        with st.spinner("Generating Heatmap..."):
                            heatmap = generate_heatmap(image, interpreter, idx)
                            
                            fig, ax = plt.subplots()
                            ax.imshow(image.resize((224, 224)))
                            ax.imshow(heatmap, cmap='jet', alpha=0.5)
                            ax.axis('off')
                            st.pyplot(fig)
                            st.caption("Red areas indicate high probability regions")

                    if enable_ai:
                        st.write("AI Doctor Prescription")
                        with st.spinner("Consulting GPT..."):
                            advice = get_gpt_advice(result_class)
                            st.info(advice)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
