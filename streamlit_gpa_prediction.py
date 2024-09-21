import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app title
st.title('Student GPA Prediction App')

# Input fields with tooltips for user data (Ethnicity and GradeClass removed)
age = st.number_input('Age', min_value=15, max_value=22, value=17, help="The student's age in years.")
gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female', 
                      help="The student's gender: 1 for Male, 0 for Female.")
parent_education = st.number_input('Parental Education Level (encoded)', min_value=0, max_value=4, value=2, 
                                   help="The highest level of education completed by the student's parents (encoded as a numerical value).")
study_time = st.number_input('Study Time Weekly (hours)', value=10.0, 
                             help="The number of hours the student studies on average per week.")
absences = st.number_input('Absences', min_value=0, max_value=50, value=5, 
                           help="The number of school days the student has been absent.")
tutoring = st.selectbox('Tutoring', options=[0, 1], 
                        help="Whether the student is receiving tutoring (1 for Yes, 0 for No).")
parent_support = st.number_input('Parental Support (encoded)', min_value=0, max_value=3, value=2, 
                                 help="The level of parental support the student receives (encoded as a numerical value).")
extracurricular = st.selectbox('Extracurricular Activities', options=[0, 1], 
                               help="Whether the student participates in extracurricular activities (1 for Yes, 0 for No).")
sports = st.selectbox('Sports Participation', options=[0, 1], 
                      help="Whether the student participates in sports (1 for Yes, 0 for No).")
music = st.selectbox('Music Participation', options=[0, 1], 
                     help="Whether the student participates in music activities (1 for Yes, 0 for No).")
volunteering = st.selectbox('Volunteering', options=[0, 1], 
                            help="Whether the student participates in volunteer work (1 for Yes, 0 for No).")

# Prepare input data for prediction (verify feature count)
input_data = np.array([[age, gender, parent_education, study_time, absences, tutoring,
                        parent_support, extracurricular, sports, music, volunteering]])

# Print shape to ensure correct number of features
print(f"Input data shape: {input_data.shape}")

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict GPA
if st.button('Predict GPA'):
    predicted_gpa = model.predict(input_data_scaled)
    st.write(f'Predicted GPA: {predicted_gpa[0]:.2f}')
