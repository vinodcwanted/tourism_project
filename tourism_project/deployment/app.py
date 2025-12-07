import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(
    repo_id="vinodcwanted/Tourism-Package-Prediction", 
    filename="best_tourism_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Product Adoption Prediction
st.title("Tourism Package Adoption Prediction App")
st.write(
    "This internal tool predicts whether a customer is likely to **take a tourism package** "
    "based on their profile and interaction details."
)
st.write("Kindly enter the customer details to check whether they are likely to take the product (ProdTaken).")

# ---------------------------
# Collect user input (tourism features)
# ---------------------------

# Numeric features
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (in minutes)", min_value=0, max_value=300, value=30)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=20, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips Taken", min_value=0, max_value=50, value=1)
Passport_input = st.selectbox("Does the customer have a Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score (1–5)", [1, 2, 3, 4, 5])
OwnCar_input = st.selectbox("Does the customer own a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)

# Categorical features
TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited", "Phone Enquiry", "Other"]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Self Employed", "Business", "Student", "Retired", "Other"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King", "Other"]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Widowed", "Other"]
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"]
)

# ---------------------------
# Build input DataFrame to match training features
# ---------------------------

input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport_input == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar_input == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# ---------------------------
# Prediction
# ---------------------------

classification_threshold = 0.45

if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "will **take** the tourism product (ProdTaken = 1)" if prediction == 1 else "is **unlikely to take** the tourism product (ProdTaken = 0)"

    st.write(f"**Predicted probability of taking the product:** {prediction_proba:.2f}")
    st.write(f"Based on the information provided, the customer {result}.")
