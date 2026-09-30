from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "skin_cancer_model.keras"
CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

st.title("Semester Project: Skin Cancer Classifier")
st.subheader("Dermoscopy Image Analysis")

uploaded_file = st.file_uploader("Upload Dermoscopic Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Dermoscopic Image", width=300)

    if model is None:
        st.warning(
            "Model not found. Train the model first with the dataset and save it to "
            f"{MODEL_PATH}."
        )
    else:
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        if st.button("Run AI Analysis"):
            predictions = model.predict(img_array, verbose=0)[0]
            max_index = int(np.argmax(predictions))
            confidence = float(predictions[max_index] * 100)

            st.success(f"Diagnosis: **{CLASS_NAMES[max_index]}**")
            st.info(f"Confidence: {confidence:.2f}%")

else:
    st.info("Upload an image to start the diagnosis.")

if model is None:
    st.caption("Tip: place the trained model at models/skin_cancer_model.keras")