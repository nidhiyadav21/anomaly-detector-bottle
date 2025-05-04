import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('model/keras_model.h5')

# ✅ Class names (Teachable Machine order: left-to-right)
class_names = ['Normal', 'Defective']  # Corrected

st.set_page_config(page_title="Bottle Defect Detector", layout="centered")
st.title("🔍 Bottle Defect Detection")
st.write("Upload an image (or use your webcam) to detect whether a bottle is **normal** or **defective**.")

# Sidebar
st.sidebar.title("👨‍🔬 Bonus Features")
use_webcam = st.sidebar.button("📷 Use Webcam")

# Webcam feature
if use_webcam:
    st.info("Webcam will open in a separate window. Press **Q** to capture a photo.")

    # Use OpenCV to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
    else:
        st.write("🔴 Webcam active... Press Q to take a snapshot.")

        while True:
            ret, frame = cap.read()
            cv2.imshow('Press Q to Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('capture.jpg', frame)
                break
        cap.release()
        cv2.destroyAllWindows()

        # Show captured image
        img = Image.open('capture.jpg').convert("RGB")
        st.image(img, caption="📷 Captured Image", use_column_width=True)

        # Preprocess and Predict
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        st.subheader(f"📢 Prediction: **{predicted_class}**")
        st.write("📊 Confidence Scores:")
        for i, score in enumerate(predictions[0]):
            st.write(f"{class_names[i]}: {score * 100:.2f}%")

        if predicted_class == 'Defective':
            st.warning("⚠️ Defect detected! Remove this bottle.")
        else:
            st.success("✅ Bottle is normal. No defect detected.")

        # 🔁 Retake option
        if st.button("🔄 Retake Photo"):
            st.experimental_rerun()







# Image upload
uploaded_file = st.file_uploader("📁 Or upload a bottle image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.subheader(f"📢 Prediction: **{predicted_class}**")
    st.write("📊 Confidence Scores:")
    for i, score in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {score * 100:.2f}%")

    if predicted_class == 'Defective':
        st.warning("⚠️ Defect detected! Remove this bottle.")
    else:
        st.success("✅ Bottle is normal. No defect detected.")

