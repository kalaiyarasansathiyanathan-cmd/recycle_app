import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# 1. Page Configuration & Setup
st.set_page_config(
    page_title="Plastic Classification & Recycling",
    page_icon="♻️",
    layout="centered"
)

MODEL_PATH = r"C:\Users\sjsrg\plastic_classifier_model.keras"

# Full names mapping
plastic_full_names = {
    "PET": "PET (Polyethylene Terephthalate)",
    "HDPE": "HDPE (High-Density Polyethylene)",
    "PVC": "PVC (Polyvinyl Chloride)",
    "LDPE": "LDPE (Low-Density Polyethylene)",
    "PP": "PP (Polypropylene)",
    "PS": "PS (Polystyrene)"
}

class_names = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

# Detailed sentence-based recycling ideas in English
recycling_ideas = {
    "PET": [
        "You can upcycle this bottle into a self-watering planter by cutting it in half and adding a cotton wick.",
        "It can be repurposed as a neat pencil holder for your study table or desk.",
        "You can turn it into a simple piggy bank by making a small slot on top for saving coins."
    ],
    "HDPE": [
        "This sturdy container can be used to store household detergents, liquids, or cleaning supplies.",
        "You can transform it into a durable watering can for your home garden by drilling small holes in the cap.",
        "It can be cut and crafted into practical desk organizers for stationary items."
    ],
    "PVC": [
        "These pipes can be converted into custom desk organizers for tools and office accessories.",
        "You can use them as flexible cable managers to neatly organize cluttered wires behind your desk.",
        "They can be used as durable border edging for garden beds and outdoor pathways."
    ],
    "LDPE": [
        "You can weave clean plastic bags together into 'plarn' (plastic yarn) to make reusable shopping bags.",
        "They can be fused together using an iron to create waterproof mats or protective covers."
    ],
    "PP": [
        "You can chop colorful caps into small tiles to create creative mosaic art projects.",
        "These containers can be cut and fitted as custom drawer dividers for small household items.",
        "They make excellent small seed starter pots for growing new plants at home."
    ],
    "PS": [
        "You can craft foam or rigid polystyrene pieces into lightweight picture frames.",
        "It can be reused to build durable architectural or hobby models for DIY projects."
    ]
}

# 2. Cache & Load Model
@st.cache_resource
def load_keras_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None

model = load_keras_model()

# 3. Streamlit UI Design
st.title("♻️ Plastic Type Classifier & Recycling Ideas")
st.write("Upload an image of a plastic item to predict its type and get recycling ideas.")

uploaded_file = st.file_uploader(
    "Choose a plastic image...", 
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Classifying plastic type..."):
        try:
            # Image Preprocessing
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            # Model Prediction
            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            predicted_short = class_names[predicted_idx]
            predicted_full = plastic_full_names.get(predicted_short, predicted_short)
            confidence = float(np.max(predictions[0])) * 100

            # Display Results
            st.success(f"**Predicted Plastic Type:** {predicted_full}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # Display Recycling Suggestions
            st.subheader(f"💡 Real-Time Recycling Ideas for {predicted_full}:")
            suggestions = recycling_ideas.get(predicted_short, ["No suggestions found."])
            
            for item in suggestions:
                st.write(f"- {item}")

        except Exception as e:
            st.error(f"Error processing image: {e}")