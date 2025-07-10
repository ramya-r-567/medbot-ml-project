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
    "Chronic cholestasis": " 🧑‍⚕️Consult Your Doctor. Check Your levels of cholesterol and certain liver enzymes.",
    "Peptic ulcer diseae": "📉 Lower Your Stomach Acid Levels.🍴 Adjust Your Meal Plan.",
    "Gastroenteritis": "🧂 Drink Fluids More Often. 😷 Stay Hygiene.",
    "Bronchial Asthma": "😷 Stay Hygiene and Away From Dust.",
    "Cervical spondylosis": "🏃‍♂️ Regular Exercise.  💆 Massage Your Neck. 🫚 Try Ginger for Relief.",
    "Paralysis (brain hemorrhage)": " 🚨Medical Emergency. Immediately Take Treatment.",
    "Jaundice": "🏃‍♂️ Exercise daily. 🥗 Healthy Diet. ❌Avoid Alcohol.",
    "hepatitis A": "🛏️ Get lots of rest. 💊Take pain relieving medication with caution.",
    "Hepatitis B": "Discuss your treatment options with your doctor 🧑‍⚕️.",
    "Hepatitis C": "🥗 Eat a well-balanced diet. 🏃 Exercise regularly. 🧪 Get tested for HIV and hepatitis B.",
    "Hepatitis D": "🧑‍⚕️ Talk to your doctor before taking prescription drugs or nutritional supplements. 🥗 Eat a well-balanced diet. 🏃 Exercise regularly.",
    "Hepatitis E": "🛏️ Rest. 🥗 Eat healthy.🧂Drink lots of water. Avoid alcohol.",
    "Tuberculosis": "🔆 Get Some Sunshine. ⚡Get Enough B-Vitamins And Iron.🥛 Drink Milk.",
    "Pneumonia": "🍵 Drink hot peppermint tea. 💊 Take an over-the-counter pain reliever. 💧Stay Hydrated.",
    "Dimorphic hemmorhoids(piles)": "❄️ Cold Compress. 🏃 Exercise. 🥗 Fibre-Rich Diet. 💧Stay Hydrated.",
    "Hyperthyroidism": "Stress Management. 🏃 Exercise regularly. 🛏️ Rest.",
    "Hypoglycemia": "🍣 Protein Snacks. Limit Refined Sugars. 🛏️ Adequate Sleep.",
    "Arthritis": "Manage your weight. 🪡Try acupuncture. 🥗Follow a healthy diet.",
    "Urinary tract infection": "😷 Maintain healthy hygiene. 🧂 Drink Fluids More Often.🫚 Consume garlic and garlic supplements.",
    "Psoriasis": " Prevent dry skin. 🙇‍♂️Reduce stress. 🥗 Eat a well-balanced diet.",
    "(vertigo) Paroymsal  Positional Vertigo": "💧Stay Hydrated. Stress Management. Improve Vitamin D Supplementation.",
    "Acne": "Apply  Apple Cider Vinegar. Take Zinc Supplements."
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
if user_input.strip() == "":
    st.info("📝 Please enter your symptoms to get a prediction.")
else:
    #Translate input to english if needed
    translated_input = translate_to_english(user_input)

    if translated_input.strip().lower() != user_input.strip().lower():
        st.info(f"🔤 Translated Input (to English): {translated_input}")

    predicted_disease = predict_disease(translated_input)

    st.subheader("😷 Predicted Disease:")
    st.success(predicted_disease)
    
    if predicted_disease in solutions:
        st.subheader("💡 Suggested Solutions:")
        st.success(solutions[predicted_disease])
    else:
        st.warning("No solution available for this disease yet.")
        


# Footer
st.markdown("---")
st.markdown("<center><small>Built with ❤️ using Streamlit & Machine Learning</small></center>", unsafe_allow_html=True)
