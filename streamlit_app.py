import streamlit as st
# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add floating blurred medical icons to background
st.markdown("""
<div class="background-icons">
  <img src="https://img.icons8.com/ios-filled/100/heart-with-pulse.png">
  <img src="https://img.icons8.com/ios-filled/100/stethoscope.png">
  <img src="https://img.icons8.com/ios-filled/100/pill.png">
  <img src="https://img.icons8.com/ios-filled/100/medical-doctor.png">
  <img src="https://img.icons8.com/ios-filled/100/first-aid-kit.png">
  <img src="https://img.icons8.com/ios-filled/100/syringe.png">
  <img src="https://img.icons8.com/ios-filled/100/dna.png">
  <img src="https://img.icons8.com/ios-filled/100/thermometer.png">
</div>
""", unsafe_allow_html=True)

import joblib
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from deep_translator import GoogleTranslator

# Load model and symptoms
model = joblib.load("last_medbot_model.pkl")
symptoms = joblib.load("last_symptom_list.pkl")

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
    "Bronchial Asthma": "😷 Stay Hygiene and Avoid Dust.",
    "Cervical spondylosis": "🏃‍♂️ Exercise Regularly. 💆 Massage Your Neck. 🫚 Try Ginger for Relief.",
    "hepatitis A": "🛏️ Get lots of rest. 💊 Take pain relievers carefully.",
    "Hepatitis B": "Discuss treatment options with your doctor 🧑‍⚕️.",
    "Hepatitis C": "🥗 Eat a balanced diet. 🏃 Exercise. 🧪 Get tested.",
    "Hepatitis D": "🧑‍⚕️ Consult before taking medications. 🥗 Eat well. 🏃 Exercise.",
    "Hepatitis E": "🛏️ Rest. 🥗 Eat healthy. 🧂saty Hydrate. ❌ Avoid alcohol.",
    "Tuberculosis": "🔆 Get sunlight. ⚡ Take B-vitamins & iron. 🥛 Drink milk.",
    "Pneumonia": "🍵 Drink hot tea. 💊 Pain relief. 💧 Hydrate.",
    "Dimorphic hemmorhoids(piles)": "❄️ Cold Compress. 🏃 Exercise. 🥗 High Fiber Diet. 💧stay Hydrate.",
    "Hyperthyroidism": "🧘 Stress Management. 🏃 Exercise. 🛏️ Rest.",
    "Hypoglycemia": "🍣 Protein Snacks. Limit Sugar. 🛏️ Sleep Well.",
    "Arthritis (common joint pain)": "⚖️ Manage weight. 🪡 Acupuncture. 🥗 Healthy diet.",
    "Urinary tract infection": "😷 Hygiene. 🧂 Hydration. 🫚 Garlic intake.",
    "Psoriasis": "🧴 Prevent dryness. 🙇‍♂️ Reduce stress. 🥗 Eat balanced meals.",
    "(vertigo) Paroymsal  Positional Vertigo": "💧Hydrate. 🙇‍♀️ Stress control. ☀️ Vitamin D.",
    "Wear and Tear(kind of joint diseases)":"🌡️Hot and Cold Therapy.🏃‍♂️Gentle Exercise.⚖️ Manage weight.",
    "Primary Headache":"💧Stay Hydrated.🛏️Rest and Relaxation.🥦Dietary Considerations.🙇‍♂️ Reduce stress.",
    "Secondary Headache":"💊 Take prescribed pain reliviers.🌡️ Temperature Therapy(Cold Pack or Warm Compress).",
    "Cluster Headache":"🫁 Breathing Exercises.❄️ Cold Compress. Avoid Triggers.",
    "Dehydration":"🧂Drink More Water.🥤Avoid Dehydrating Beverages. Eat Water-Rich Foods.",
    "Bacterial Skin Infection":"🫧Clean the sores gently.Keep sores covered.😷 Stay Hygiene.",
    "Menstruation pain":"🌡️Heat Therapy.🌿Herbal Teas and Supplements.💧Stay Hydrated.",
    "GRED(chronic digestive condition )":"🥦 Dietary and Eating Habit Changes.🧘 Stress Management.💪Maintain a Healthy Weight.",
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
}

selected_lang_label = st.selectbox("🌐 Select Language / மொழியை தேர்ந்தெடுக்கவும்:", list(language_map.keys()))
selected_lang = language_map[selected_lang_label]

# Translated UI Text
title = translate("🤖 MedBot AI – Your Symptom Checker", "en", selected_lang)
symptom_label = translate("Describe your symptoms in any language:", "en", selected_lang)
predicted_disease_label = translate("😷 Predicted Disease:", "en", selected_lang)
suggested_solution_label = translate("💡 Suggested Solution:", "en", selected_lang)
empty_input_info = translate("📝 Please enter your symptoms to get a prediction.", "en", selected_lang)
no_solution_text = translate("No solution available for this disease yet.", "en", selected_lang)

# Title
st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)

# Select from list (only mode)
symptoms_translated = [translate(sym, "en", selected_lang) for sym in symptoms]
selected_symptoms = st.multiselect(
    translate("Select your symptoms", "en", selected_lang),
    symptoms_translated
)

if st.button(translate("Predict from Selection", "en", selected_lang)):
    if selected_symptoms:
        # Translate back to English for model input
        selected_symptoms_en = [translate(sym, selected_lang, "en") for sym in selected_symptoms]
        features = [1 if s in selected_symptoms_en else 0 for s in symptoms]
        prediction = model.predict(np.array(features).reshape(1, -1))[0]

        matched_key = next((key for key in solutions if key.lower() == prediction.strip().lower()), None)

        st.subheader(predicted_disease_label)
        st.success(translate(prediction, 'en', selected_lang))

        if matched_key:
            st.subheader(suggested_solution_label)
            st.success(translate(solutions[matched_key], 'en', selected_lang))
        else:
            st.warning(no_solution_text)
    else:
        st.info(translate("📝 Please select at least one symptom.", "en", selected_lang))

# Add floating blurred medical icons to background
st.markdown("""
<div class="background-icons">
  <img src="https://img.icons8.com/ios-filled/100/heart-with-pulse.png">
  <img src="https://img.icons8.com/ios-filled/100/stethoscope.png">
  <img src="https://img.icons8.com/ios-filled/100/pill.png">
  <img src="https://img.icons8.com/ios-filled/100/medical-doctor.png">
  <img src="https://img.icons8.com/ios-filled/100/first-aid-kit.png">
  <img src="https://img.icons8.com/ios-filled/100/syringe.png">
  <img src="https://img.icons8.com/ios-filled/100/dna.png">
  <img src="https://img.icons8.com/ios-filled/100/thermometer.png">
</div>
""", unsafe_allow_html=True)
