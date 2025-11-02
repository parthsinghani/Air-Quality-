import streamlit as st
import joblib
import pandas as pd
import time

# --- Page Configuration (Adds "Glamour"!) ---
st.set_page_config(
    page_title="HealthScan AI Predictor",
    page_icon="🧬",
    layout="wide"
)

# --- Load Our Trained Model ---
# We load the model we saved from the 'train_model.py' script
try:
    model = joblib.load('disease_predictor_model.joblib')
    print("Model loaded successfully")
except FileNotFoundError:
    st.error("Model file not found! Please run 'train_model.py' first to create it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- The App Title ---
st.title("HealthScan AI: Disease Likelihood Predictor 🩺")
st.markdown("Enter your symptoms below, and our AI will provide a potential diagnosis. *This is a demo and not for medical use.*")


# --- Create Columns for Layout ---
# We can use columns to make it look cleaner
col1, col2 = st.columns([1, 2]) # 1/3 for inputs, 2/3 for results


# --- Input Fields (in the first column) ---
with col1:
    st.header("Your Symptoms 🤒")
    
    # Slider for Fever
    temp = st.slider("What is your body temperature (°F)?", 97.0, 105.0, 98.6, 0.1)
    
    # Selectbox for Cough
    cough = st.selectbox("Do you have a cough?", ("No", "Yes"))
    
    # Selectbox for Fatigue
    fatigue = st.selectbox("Are you experiencing fatigue?", ("No", "Yes"))
    
    # Selectbox for Runny Nose
    runny_nose = st.selectbox("Do you have a runny nose?", ("No", "Yes"))

    # The Big "Predict" Button
    predict_button = st.button("✨ Predict My Condition", type="primary")


# --- Convert Yes/No to 1/0 for the model ---
cough_val = 1 if cough == "Yes" else 0
fatigue_val = 1 if fatigue == "Yes" else 0
runny_nose_val = 1 if runny_nose == "Yes" else 0

# --- Create the input DataFrame for the model ---
# This MUST match the features the model was trained on
input_data = pd.DataFrame({
    'Fever': [temp],
    'Cough': [cough_val],
    'Fatigue': [fatigue_val],
    'RunnyNose': [runny_nose_val]
})


# --- The Prediction Logic (in the second column) ---
with col2:
    st.header("Analysis Results 🔬")
    
    if predict_button:
        # Show a "loading" spinner for dramatic effect
        with st.spinner('Our AI is analyzing your symptoms...'):
            time.sleep(1) # Just for a cool effect
            
            # Make the prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            # Get the predicted disease
            disease = prediction[0]
            
            # Get the confidence score
            confidence = prediction_proba.max() * 100
            
            st.success(f"Prediction Complete! Confidence: {confidence:.0f}%")
            
            # This is the "glamorous" output!
            if disease == 'Flu':
                st.metric(label="Predicted Condition", value="Influenza (Flu) 🤧", delta="High Severity")
                st.warning("You show strong symptoms of the Flu. Please rest and consult a doctor.", icon="⚠️")
            elif disease == 'Cold':
                st.metric(label="Predicted Condition", value="Common Cold 🥶", delta="Low Severity")
                st.info("Looks like a Common Cold. Get some rest and drink fluids.", icon="ℹ️")
            else: # Allergy
                st.metric(label="Predicted Condition", value="Allergies 🌷", delta="Low Severity")
                st.info("Your symptoms are consistent with allergies. Antihistamines may help.", icon="ℹ️")

            # This adds the "glam" you wanted!
            st.balloons()
            
            # Show the probabilities in a nice chart
            st.subheader("Prediction Breakdown:")
            st.bar_chart(pd.DataFrame(prediction_proba, columns=model.classes_), x=None, y=None)

    else:
        st.info("Please enter your symptoms and click the 'Predict' button.")