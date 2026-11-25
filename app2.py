import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# ---------- Config ----------
CLASSIFIER_PATH = "model.h5"   # image classification model
YOLO_PATH = r"D:\GUVI\Shore_sense\models\best.pt"          # object detection model

classes = [
    "bottlenose_dolphin",
    "green_turtle",
    "hammerhead_shark",
    "hawaiian_monk_seal",
    "north_atlantic_right_whale",
    "whale_shark",
]

# ---------- Load models (cached) ----------
@st.cache_resource
def load_classifier():
    if not os.path.exists(CLASSIFIER_PATH):
        st.error(f"Classifier not found at {CLASSIFIER_PATH}")
        return None
    return load_model(CLASSIFIER_PATH)

@st.cache_resource
def load_detector():
    if not os.path.exists(YOLO_PATH):
        st.error(f"YOLO model not found at {YOLO_PATH}")
        return None
    return YOLO(YOLO_PATH)

classifier = load_classifier()
yolo = load_detector()

# ---------- Helper functions ----------
def classify_species(image):
    img = image.convert("RGB").resize((224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = classifier.predict(arr, verbose=0)[0]
    return preds

def detect_species(image):
    result = yolo.predict(image, conf=0.25, verbose=False)[0]
    if len(result.boxes) == 0:
        return None, None, None
    i = int(result.boxes.conf.argmax())
    species = yolo.names[int(result.boxes.cls[i])]
    confidence = float(result.boxes.conf[i])
    plotted = result.plot()[:, :, ::-1]  # BGR -> RGB
    return species, confidence, plotted

# ---------- UI ----------
st.title("🐋 Endangered Marine Species Classification & Detection")
st.write("Upload an image of a marine animal to classify and detect it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and classifier is not None and yolo is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    

    # Classification
    
    st.subheader("🔍 Classification")
    with st.spinner("Classifying..."):
        preds = classify_species(image)
        pred_class = classes[int(np.argmax(preds))]
        st.success(f"Species: {pred_class}")
        st.info(f"Confidence: {np.max(preds) * 100:.2f}%")

    with st.expander("All class probabilities"):
            for cls, prob in zip(classes, preds):
                st.write(f"{cls}: {prob * 100:.2f}%")
                st.progress(float(prob))

    # Detection
    
    st.subheader("📦 Detection")
    with st.spinner("Detecting..."):
            species, conf, plotted = detect_species(image)
    if species is None:
            st.warning("No animal detected.")
    else:
            st.success(f"Species: {species}")
            st.info(f"Confidence: {conf * 100:.2f}%")
            st.image(plotted, caption="Detection Result", use_container_width=True)