import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
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

def analyze_quadrants(image, interpreter, class_names):
    w, h = image.size
    mid_w, mid_h = w // 2, h // 2
    
    crops = {
        "Top-Left": image.crop((0, 0, mid_w, mid_h)),
        "Top-Right": image.crop((mid_w, 0, w, mid_h)),
        "Bottom-Left": image.crop((0, mid_h, mid_w, h)),
        "Bottom-Right": image.crop((mid_w, mid_h, w, h))
    }
    
    results = {}
    
    for name, crop_img in crops.items():
        preds = predict_image(crop_img, interpreter)
        idx = np.argmax(preds)
        conf = np.max(preds) * 100
        pred_class = class_names[idx]
        
        results[name] = {
            "image": crop_img,
            "class": pred_class,
            "confidence": conf
        }
        
    return results

def get_gpt_advice(detected_issues, location):
    issues_str = ", ".join(detected_issues)
    if "Healthy" in issues_str and len(detected_issues) == 1:
        return "The plant looks healthy in all examined areas. Maintain regular care."
    
    try:
        api_key = st.secrets["openai_key"]
    except:
        return "OpenAI API Key missing in Secrets."

    client = openai.OpenAI(api_key=api_key)
    
    loc_text = f"in {location}" if location else ""
    
    prompt = f"""
    You are an expert Agronomist. A farmer {loc_text} scanned a crop and found the following issues in different parts of the plant: {issues_str}.
    Provide a combined recommendation:
    1. Analysis of the mix (e.g., if both weeds and disease are present).
    2. Immediate Cure (Chemical or Organic) for each distinct issue.
    3. Prevention for next season.
    Keep it concise and actionable.
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
                    
                    main_preds = predict_image(image, interpreter)
                    main_idx = np.argmax(main_preds)
                    main_conf = np.max(main_preds) * 100
                    main_result = current_classes[main_idx]
                    
                    st.success(f"Overall Diagnosis: {main_result}")
                    st.metric("Overall Confidence", f"{main_conf:.2f}%")
                    
                    unique_detected_issues = {main_result}

                    if enable_xai:
                        st.write("### Multi-Zone Analysis")
                        st.caption("Independent analysis of each image quadrant:")
                        
                        with st.spinner("Analyzing quadrants..."):
                            quad_results = analyze_quadrants(image, interpreter, current_classes)
                            
                            row1 = st.columns(2)
                            row2 = st.columns(2)
                            
                            def display_quad(col, title, data):
                                with col:
                                    st.image(data["image"], use_column_width=True)
                                    color = ":red" if "Healthy" not in data["class"] else ":green"
                                    st.markdown(f"**{title}**")
                                    st.markdown(f"{color}[{data['class']}] ({data['confidence']:.1f}%)")
                                    return data["class"]

                            c1 = display_quad(row1[0], "Top-Left", quad_results["Top-Left"])
                            c2 = display_quad(row1[1], "Top-Right", quad_results["Top-Right"])
                            c3 = display_quad(row2[0], "Bottom-Left", quad_results["Bottom-Left"])
                            c4 = display_quad(row2[1], "Bottom-Right", quad_results["Bottom-Right"])
                            
                            unique_detected_issues.update([c1, c2, c3, c4])

                    if enable_ai:
                        st.write("### AI Doctor Prescription")
                        
                        final_issue_list = list(unique_detected_issues)
                        if "Healthy" in final_issue_list and len(final_issue_list) > 1:
                            final_issue_list = [i for i in final_issue_list if "Healthy" not in i]
                            
                        with st.spinner("Consulting GPT..."):
                            advice = get_gpt_advice(final_issue_list, user_location)
                            st.info(advice)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
