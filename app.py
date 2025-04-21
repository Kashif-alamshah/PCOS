import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from langchain_advisor import LangchainHealthAdvisor

#-----------------------------------------Load model once-------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pcos_model.h5")

model = load_model()

# Constants
optimal_threshold = 0.828086256980896

# -----------------------------------------Initialize LangchainHealthAdvisor------------------------------------------
advisor = LangchainHealthAdvisor()

# -----------------------------------------Center-align title and description-----------------------------------------
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>🧫 Polycystic Ovary Syndrome (PCOS) Image Classifier</h1>
        <hr style='border: 1px solid #bbb; width: 80%; margin: auto;'>
        <p>Upload an Ultrasound image to classify as <strong>Infected</strong> or <strong>Non-Infected</strong>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add spacing
st.markdown("###")
left, right = st.columns([1, 1])

with left:
    uploaded_file = st.file_uploader("📤 Upload a cell image", type=["jpg", "jpeg", "png"])

    def predict_image(img_path_or_file, model, threshold=optimal_threshold):
        img = image.load_img(img_path_or_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction_score = model.predict(img_array)[0][0]
        if prediction_score > threshold:
            result = "Non-Infected"
            st.success("🟢 **Prediction: Non-Infected**")
        else:
            result = "Infected"
            st.error("🔴 **Prediction: Infected**")
        suggestions = advisor.get_advice(prompt=f"Provide health advice for someone with PCOS who is {result}.")
        st.markdown("### 💡 AI-Generated Health Suggestions")
        st.info(suggestions)

    if uploaded_file is not None:
        predict_image(uploaded_file, model)

with right:
    if uploaded_file is not None:
        st.image(uploaded_file, caption="🖼️ Uploaded Image", use_container_width=True)
