import streamlit as st
import joblib
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from deep_translator import GoogleTranslator


# Load model and symptoms
model = joblib.load("medbot_model.pkl")
symptoms = joblib.load("symptom_list.pkl")

# Dictionary mapping diseases to simple solutions
solutions = {
    "Fungal infection": "🧴 Use antifungal cream. Keep the area clean and dry.",
    "Allergy": "💊 Take antihistamines. Avoid allergens.",
    "GERD": "🍽️ Eat smaller meals. Avoid spicy food. Try antacids.",
    "Diabetes": "🥗 Eat healthy. 🏃 Exercise regularly. 🩺 Visit an endocrinologist.",
    "Hypertension": "🧘‍♀️ Reduce salt. 🏃‍♂️ Exercise daily. Take prescribed meds.",
    "Migraine": "💆‍♀️ Rest in a dark room. Take migraine meds.",
    "Chickenpox": "🛏️ Rest. Calamine lotion for itching. Stay hydrated.",
    "AIDS": "🧑‍⚕️ Take a Prescribed ART(antiretroviral therapy).",
    "Jaundice": "💧Stay Hydrated. 🥗 Dietary Adjustment. 🛏️Take rest.",
    "Malaria": "🧑‍⚕️ Consult Doctor Immediately.",
    "Dengue": "💧Stay Hydrated.🛏️ Take Plenty of Rest.",
    "Typhoid": "Take Antibiotics Prescribed by Doctor 🧑‍⚕️.",
    "Common Cold": "🛏️ Rest.💧Stay Hydrated. Gargling With Warm Water."
    # Add more if needed
}

# Set up vectorizer to process free text input
vectorizer = CountVectorizer(vocabulary=symptoms)

def preprocess_input(text):
    # Clean and lowercase the text
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    return text


def translate_to_english(text):
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    return translated


def predict_disease(user_input):
    # Translate input first
    translated_input = translate_to_english(user_input)
    cleaned_text = preprocess_input(translated_input)

    # Optional: show translated input
    st.write("🔤 Translated Input (to English):", translated_input)

    vector = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(vector)
    return prediction[0]

# Streamlit UI starts here
st.set_page_config(page_title="MedBot AI", page_icon="💊", layout="centered")

# Background color using HTML/CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f2ff;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .title {
        color: #004d99;
        font-size: 42px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<div class='title'>🤖 MedBot AI – Your Symptom Checker</div>", unsafe_allow_html=True)
st.write("\n")

# User input for symptoms
user_input = st.text_input("Describe your symptoms in natural language (e.g., 'I have fever and headache'):", key="user_input")

# Predict on pressing enter
if user_input:
    predicted_disease = predict_disease(user_input)
    st.subheader("🤖 Predicted Disease:")
    st.success(predicted_disease)

    # Display suggested solution
    if predicted_disease in solutions:
        st.subheader("💡 Suggested Solution:")
        st.success(solutions[predicted_disease])
    else:
        st.info("No solution available for this disease yet.")

# Footer
st.markdown("---")
st.markdown("<center><small>Built with ❤️ using Streamlit & Machine Learning</small></center>", unsafe_allow_html=True)
