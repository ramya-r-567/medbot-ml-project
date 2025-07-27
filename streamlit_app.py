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
    "Common Cold": "🛏️ Rest.💧Stay Hydrated. Gargling With Warm Water.",
    "Chronic cholestasis": "🧑‍⚕️ Consult Your Doctor. Check Your cholesterol & liver enzyme levels.",
    "Peptic ulcer diseae": "📉 Lower Your Stomach Acid Levels.🍴 Adjust Your Meal Plan.",
    "Gastroenteritis": "🧂 Drink Fluids More Often. 😷 Stay Hygienic.",
    "Bronchial Asthma": "😷 Stay Hygienic and Avoid Dust.",
    "Cervical spondylosis": "🏃‍♂️ Exercise Regularly. 💆 Massage Your Neck. 🫚 Try Ginger for Relief.",
    "Paralysis (brain hemorrhage)": "🚨 Medical Emergency. Seek Immediate Treatment.",
    "hepatitis A": "🛏️ Get lots of rest. 💊 Take pain relievers carefully.",
    "Hepatitis B": "Discuss treatment options with your doctor 🧑‍⚕️.",
    "Hepatitis C": "🥗 Eat a balanced diet. 🏃 Exercise. 🧪 Get tested.",
    "Hepatitis D": "🧑‍⚕️ Consult before taking medications. 🥗 Eat well. 🏃 Exercise.",
    "Hepatitis E": "🛏️ Rest. 🥗 Eat healthy. 🧂 Hydrate. ❌ Avoid alcohol.",
    "Tuberculosis": "🔆 Get sunlight. ⚡ Take B-vitamins & iron. 🥛 Drink milk.",
    "Pneumonia": "🍵 Drink hot tea. 💊 Pain relief. 💧 Hydrate.",
    "Dimorphic hemmorhoids(piles)": "❄️ Cold Compress. 🏃 Exercise. 🥗 High Fiber Diet. 💧Hydrate.",
    "Hyperthyroidism": "🧘 Stress Management. 🏃 Exercise. 🛏️ Rest.",
    "Hypoglycemia": "🍣 Protein Snacks. Limit Sugar. 🛏️ Sleep Well.",
    "Arthritis": "⚖️ Manage weight. 🪡 Acupuncture. 🥗 Healthy diet.",
    "Urinary tract infection": "😷 Hygiene. 🧂 Hydration. 🫚 Garlic intake.",
    "Psoriasis": "🧴 Prevent dryness. 🙇‍♂️ Reduce stress. 🥗 Eat balanced meals.",
    "(vertigo) Paroymsal  Positional Vertigo": "💧Hydrate. 🙇‍♀️ Stress control. ☀️ Vitamin D.",
    "Acne": "🍎 Apple cider vinegar. 🔩 Zinc supplements."
}

vectorizer = CountVectorizer(vocabulary=symptoms)

def preprocess_input(text):
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

def translate(text, src_lang, tgt_lang):
    return GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)

def predict_disease(user_input, selected_lang):
    translated_input = translate(user_input, 'auto', 'en')  # Auto-detect source language
    cleaned = preprocess_input(translated_input)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    return prediction


# Streamlit UI Config
st.set_page_config(page_title="MedBot AI", page_icon="💊", layout="centered")

# Language Selection
language_map = {
    "English": "en",
    "தமிழ் (Tamil)": "ta",
    "हिन्दी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "ಕನ್ನಡ (Kannada)": "kn",
    "മലയാളം (Malayalam)": "ml",
    "বাংলা (Bengali)": "bn"
}

selected_lang_label = st.selectbox("🌐 Select Language / மொழியை தேர்ந்தெடுக்கவும்:", list(language_map.keys()))
selected_lang = language_map[selected_lang_label]

# Translated Titles
title = translate("🤖 MedBot AI – Your Symptom Checker", "en", selected_lang)
symptom_label = translate("Describe your symptoms in any language:", "en", selected_lang)
predicted_disease_label = translate("😷 Predicted Disease:", "en", selected_lang)
suggested_solution_label = translate("💡 Suggested Solution:", "en", selected_lang)
empty_input_info = translate("📝 Please enter your symptoms to get a prediction.", "en", selected_lang)
no_solution_text = translate("No solution available for this disease yet.", "en", selected_lang)

# Page Style and Title
st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown(f"<div class='title'>{title}</div>", unsafe_allow_html=True)
st.write("")

# User Input
user_input = st.text_input(symptom_label, key="user_input")

# Prediction
if user_input.strip():
    prediction = predict_disease(user_input)
    st.subheader(predicted_disease_label)
    st.success(translate(prediction, 'en', selected_lang))

    if prediction in solutions:
        st.subheader(suggested_solution_label)
        st.success(translate(solutions[prediction], 'en', selected_lang))
    else:
        st.warning(no_solution_text)
else:
    st.info(empty_input_info)

