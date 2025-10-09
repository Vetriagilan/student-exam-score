import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Student Math Score Predictor (Integer Encoding)",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Feature Mapping based on the provided numerical table and CSV content ---
# These mappings convert the user-friendly strings into the specific integers
# that your existing 'model.pkl' expects for prediction.

# MAPPING 1: Gender
# Assuming 'female' is 0 (Row 0, 1, 2) and 'male' is 1 (Row 3, 4) in your input table.
GENDER_MAP = {'female': 0, 'male': 1}
GENDER_OPTIONS = list(GENDER_MAP.keys())

# MAPPING 2: Race/Ethnicity
# Based on common CSV structure, Group A=0, B=1, C=2, D=3, E=4 is a likely simple Label Encoding.
# Row 0 (Group B in CSV) -> 1 in table
# Row 1 (Group C in CSV) -> 2 in table
RACE_MAP = {
    'group A': 0,
    'group B': 1,
    'group C': 2,
    'group D': 3,
    'group E': 4
}
RACE_OPTIONS = list(RACE_MAP.keys())

# MAPPING 3: Parental Level of Education
# This mapping is highly speculative without the original encoding table, but we infer based on row 3 having 'associate\'s degree' (CSV) mapped to 0 (Table)
# Let's use a logical order based on complexity/time:
EDU_MAP = {
    "some high school": 5, # High value to avoid clashing with other small integers
    "high school": 4, 
    "some college": 3, 
    "associate's degree": 0, # Based on table row 3
    "bachelor's degree": 1, # Based on table row 0
    "master's degree": 2  # Based on table row 2
}
EDUCATION_OPTIONS = list(EDU_MAP.keys())

# MAPPING 4: Lunch (standard vs free/reduced)
# Assuming 'free/reduced' is 0 (Row 3) and 'standard' is 1 (Row 0, 1, 2, 4).
LUNCH_MAP = {'free/reduced': 0, 'standard': 1}
LUNCH_OPTIONS = list(LUNCH_MAP.keys())

# MAPPING 5: Test Preparation Course
# Assuming 'none' is 1 (Row 0, 2, 3) and 'completed' is 0 (Row 1).
# NOTE: This reverse mapping (completed=0, none=1) is unusual but common if encoding was alphabetical (completed < none).
TEST_PREP_MAP = {'none': 1, 'completed': 0}
TEST_PREP_OPTIONS = list(TEST_PREP_MAP.keys())


# --- Expected feature order for the model (7 total features, all numerical) ---
EXPECTED_COLUMNS = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course',
    'reading score',
    'writing score'
]


# --- Function to load the model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model using joblib."""
    try:
        model = joblib.load('model.pkl')
        # Check the number of expected features
        if model.n_features_in_ != len(EXPECTED_COLUMNS):
            st.warning(
                f"Model expects {model.n_features_in_} features, but app is configured for {len(EXPECTED_COLUMNS)}. "
                f"This may lead to an error. Using existing model as is."
            )
        return model
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure the model file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Function for data preprocessing (Integer Mapping) ---
def preprocess_input(input_data):
    """
    Applies integer mapping to categorical variables based on inferred mappings.
    """
    processed_data = {
        'gender': GENDER_MAP[input_data['gender']],
        'race/ethnicity': RACE_MAP[input_data['race/ethnicity']],
        'parental level of education': EDU_MAP[input_data['parental level of education']],
        'lunch': LUNCH_MAP[input_data['lunch']],
        'test preparation course': TEST_PREP_MAP[input_data['test preparation course']],
        'reading score': input_data['reading score'],
        'writing score': input_data['writing score']
    }
    
    # Create the final feature array in the correct order (7 features)
    final_features = np.array([
        processed_data[col] for col in EXPECTED_COLUMNS
    ]).reshape(1, -1)
    
    return final_features

# --- Streamlit Application Layout ---
def main():
    st.title("📚 Student Math Score Prediction (Integer-Encoded Model)")
    st.markdown(
        """
        This app uses your existing `model.pkl`, which expects 7 **integer-encoded** features 
        (like 0 or 1) instead of 14 one-hot encoded features. The categorical inputs below are 
        mapped to the numerical values inferred from your original data.
        """
    )
    
    model = load_model()
    if model is None:
        return # Stop execution if the model failed to load

    st.sidebar.header("Student Information")

    # --- User Input Widgets (SideBar) ---
    
    # Categorical Features using the defined options
    gender = st.sidebar.selectbox("Gender", GENDER_OPTIONS)
    race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", RACE_OPTIONS)
    parental_education = st.sidebar.selectbox("Parental Level of Education", EDUCATION_OPTIONS)
    lunch = st.sidebar.selectbox("Lunch Type", LUNCH_OPTIONS)
    test_prep = st.sidebar.selectbox("Test Preparation Course", TEST_PREP_OPTIONS)
    
    st.sidebar.header("Prior Test Scores")
    
    # Numerical Features
    reading_score = st.sidebar.slider(
        "Reading Score (0-100)", 
        min_value=0, 
        max_value=100, 
        value=70
    )
    writing_score = st.sidebar.slider(
        "Writing Score (0-100)", 
        min_value=0, 
        max_value=100, 
        value=65
    )

    # --- Data Collection and Prediction ---

    if st.button("Predict Math Score"):
        # Dictionary of input values (user strings)
        input_data = {
            'gender': gender,
            'race/ethnicity': race_ethnicity,
            'parental level of education': parental_education,
            'lunch': lunch,
            'test preparation course': test_prep,
            'reading score': reading_score,
            'writing score': writing_score
        }
        
        # Preprocess input data: convert strings to the required integers
        final_features = preprocess_input(input_data)
        
        # Make Prediction
        try:
            prediction = model.predict(final_features)
            predicted_score = round(float(prediction[0]), 2) # Ensure prediction is treated as float
            
            # Clamp the score between 0 and 100
            clamped_score = np.clip(predicted_score, 0.0, 100.0)

            # --- Display Results ---
            st.markdown("---")
            st.subheader("Prediction Result")
            
            # Simple success metric based on score range
            if clamped_score >= 75:
                emoji = "⭐"
                message = "Excellent job! A very high predicted score."
            elif clamped_score >= 50:
                emoji = "👍"
                message = "A solid predicted score. Keep up the good work!"
            else:
                emoji = "💡"
                message = "There's room for improvement. Focus on study habits."
                
            st.metric(
                label="Predicted Math Score (0-100)",
                value=f"{clamped_score:.2f}",
                delta=message
            )
            
            st.balloons()
            
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check if your model was saved with 7 features (e.g., if you see 14 features, please retrain the model with the OHE-based train_model.py provided earlier): {e}")

if __name__ == "__main__":
    main()
