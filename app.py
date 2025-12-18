import streamlit as st
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Load models
yolo_model = YOLO("yolov8n.pt")
intent_model = load_model("models/intent_lstm.h5")
cnn_model = load_model("models/cnn_classifier.h5")

# Intent tokenizer
intent_texts = [
    "explain this",
    "summarize this",
    "give me a quiz",
    "explain for beginners",
    "explain in detail"
]

intent_tokenizer = Tokenizer(num_words=1000)
intent_tokenizer.fit_on_texts(intent_texts)

intent_map = {
    0: "Explain",
    1: "Summary",
    2: "Quiz",
    3: "Beginner",
    4: "Advanced"
}

# Streamlit UI
st.set_page_config(page_title="AI Learning Companion", layout="wide")
st.title("📘 AI Powered Learning Companion")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

user_query = st.text_input("What do you want?", "explain this")

if uploaded_file:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)

    st.image(image, caption="Uploaded Image", width=400)

    # YOLO Detection
    results = yolo_model(image)
    annotated = results[0].plot()
    st.image(annotated, caption="YOLO Detection")

    # OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray)

    if extracted_text.strip() == "":
        extracted_text = "No readable text found in the image."

    st.subheader("📄 Extracted Text")
    st.write(extracted_text)

    # CNN Content Classification
    resized = cv2.resize(image, (128,128)) / 255.0
    resized = np.expand_dims(resized, axis=0)

    cnn_pred = np.argmax(cnn_model.predict(resized, verbose=0))
    content_type = ["Text", "Diagram", "Mixed"][cnn_pred]

    st.subheader("🧠 Content Type")
    st.success(content_type)

    # Intent Detection
    seq = intent_tokenizer.texts_to_sequences([user_query])
    padded = pad_sequences(seq, maxlen=5)

    intent_pred = np.argmax(intent_model.predict(padded, verbose=0))
    intent = intent_map[intent_pred]

    st.subheader("🎯 User Intent")
    st.info(intent)

    # Response Engine
    st.subheader("🤖 AI Response")
    if intent == "Explain":
        response = f"The content is primarily {content_type.lower()}. Here's a detailed explanation of the extracted text: {extracted_text}"
    elif intent == "Summary":
        response = f"Here's a summary of the extracted text: {extracted_text[:150]}..."
    elif intent == "Quiz":
        response = "Quiz feature is under development."
    elif intent == "Beginner":
        response = f"This content is suitable for beginners. Here's a simple explanation: {extracted_text}"
    elif intent == "Advanced":
        response = f"This content is for advanced learners. Here's an in-depth analysis: {extracted_text}"
    else:
        response = "I'm not sure how to help with that."
    st.write(response)
