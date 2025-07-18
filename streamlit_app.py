import streamlit as st
import pickle 
from deep_translator import GoogleTranslator



# Load model and symptoms
model = pickle.load(open("medbot_model.pkl", "rb"))
symptoms = pickle.load(open("symptom_list.pkl", "rb"))

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
    "Acne": "Apply  Apple Cider Vinegar. Take Zinc Supplements.",
    "Diabetes": "🥗 Eat healthy. 🏃 Exercise regularly. 🩺 Visit an endocrinologist."
    # Add more if needed
}

#list of supported languages
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn"
    
}

#streamlit UI
st.set_page_config(page_title= "MedBot AI", layout="centered")
st.title("🩺 MedBot - Your Symptoms Checker")

#select language
selected_language = st.selectbox("🌐 Choose Your Language:", list(language_options.keys()))
target_lang = language_options[selected_language]

#input symptoms
user_input = st.text_input("📝 Enter Your Symptoms (comma separated):")

#predict button
if st.button("Predict Disease"):
    if user_input:
        #translate input to english if needed
        if target_lang != "en":
            translate_input = GoogleTransaltor(source= 'auto', target= 'en').translate(user_input)
        else:
            translated_input = user_input

        #convert symptoms into input format for model
        input_symptoms = [sym.strip().capitalize() for sym in translated_input.split(',')]
        input_vector = [1 if symptom in input_symptoms else 0 for symptoms in symptom_list]

        #predict
        predicted_disease = model.predict([input_vector])[0]

        #translate outputs if needed
        if target_lang != "en":
            translated_disease = GoogleTranslator(source= 'en', target=target_lang).translate(predicted_disease)
            translated_solution = GoogleTranslator(source= 'en', target=target_lang).translate(solutions.get(predicted_disease, "No solution available for this disease yet.")
            )
        else:
            translated_disease = predicted_disease
            translated_solution = solutions.get(predicted_disease, "No solution available for this disease yet")

        #display results
        st.subheader("😷 Predicted Disease:")
        st.success(translated_disease)

        st.subheader("💡Suggested Solution:")
        st.success(translated_solution)

    else:
        st.warning("Please enter your symptoms.")





