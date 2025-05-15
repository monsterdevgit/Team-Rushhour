import streamlit as st
import numpy as np
import pickle  

# Load pre-trained model and scaler
@st.cache_resource
def load_model():
    with open("loan_approval_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)  
    return model

# Predict loan eligibility
def predict_eligibility(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    
    prediction = model.predict(input_data)[0]  # 0 or 1 for ineligible or eligible
    probability = model.predict_proba(input_data)[0][1]  # Probability for eligible
    return prediction, probability

# Streamlit app
st.set_page_config(page_title="Agrolift Loan Eligibility", page_icon=":bank:", layout="centered")

# App title
st.title("Agrolift: Loan Eligibility Prediction App")

# Description
st.markdown("""
    **Welcome to Agrolift's Loan Eligibility Prediction App!**""")

# Sidebar for user inputs with better grouping
st.sidebar.header("Enter Your Details")

st.sidebar.subheader("Demographics")
Age = st.sidebar.number_input("Age", min_value=18, max_value=50, value=30)
Years_in_Farming = st.sidebar.number_input("Years in Farming", min_value=0, max_value=20, value=5)
Cooperative_Member = st.sidebar.selectbox("Cooperative Member", ["True", "False"])

st.sidebar.subheader("Skills & Training")
Digital_Literacy_Level = st.sidebar.selectbox("Digital Literacy Level", ["Low", "Medium", "High"])
Agribusiness_Training = st.sidebar.selectbox("Agribusiness Training", ["True", "False"])

st.sidebar.subheader("Farming Infrastructure")
Crop_Type = st.sidebar.selectbox("Crop Type", ['Crop Farming', 'Mixed Farming', 'Livestock Farming'])
Proximity_to_Roads = st.sidebar.selectbox("Proximity to Roads", ["Close", "Far"])
Access_to_Market = st.sidebar.selectbox("Access to Market", ["Good", "Poor"])

st.sidebar.subheader("Financial Information")
Loan_Purpose = st.sidebar.selectbox("Loan Purpose", ['Fertilizer', 'Land Expansion', 'Labor', 'Irrigation', 'Equipment'])
Repayment_Status = st.sidebar.selectbox("Repayment Status", ["Fully Paid", "Defaulted", "Partially Paid", "None"])


# Encoding categorical variables
digital_literacy = {"Low": 0, "Medium": 1, "High": 2}[Digital_Literacy_Level]
cooperative_member_encoded = 1 if Cooperative_Member == "True" else 0
agribusiness_training_encoded = 1 if Agribusiness_Training == "True" else 0
crop_type_encoded = {"Crop Farming": 0, "Livestock Farming": 1, "Mixed Farming": 2}[Crop_Type]
access_to_market_encoded = {"Good": 0, "Poor": 1}[Access_to_Market]
proximity_to_roads_encoded = {"Close": 0, "Far": 1}[Proximity_to_Roads]
loan_purpose_encoded = {"Equipment": 0, "Fertilizer": 1, "Irrigation": 2, "Labor": 3, "Land Expansion": 4}[Loan_Purpose]
repayment_status_encoded = {"Fully Paid": 3, "Defaulted": 0, "Partially Paid": 2, "None": 1}[Repayment_Status]

# Combine all inputs into a single list with matching column names
input_data = [
    Age,
    Years_in_Farming,
    digital_literacy,
    cooperative_member_encoded,
    agribusiness_training_encoded,
    crop_type_encoded,
    access_to_market_encoded,
    proximity_to_roads_encoded,
    loan_purpose_encoded,
    repayment_status_encoded,
]

# Display the inputs for verification
#st.write("Input Data:", input_data)

# Load model
model = load_model()

# Predict loan eligibility
if st.button("Predict Loan Eligibility", use_container_width=True):
    prediction, probability = predict_eligibility(model, input_data)

    # Display results with improved visuals
    if prediction == 0:
        st.success(f"🎉 Congratulations! You are **eligible** for the loan with a probability of **{probability:.2%}**.")
        st.markdown("""
        #### Next Steps:
        - Our loan officer will contact you shortly for verification and documentation.
        - Ensure you have your identification and any relevant farm or financial records ready.
        - You will receive a notification once your loan approval is finalized.

        Thank you for applying. We look forward to supporting your farming journey! 🌾
        """)
    else:
        st.error(f"Unfortunately, you are **not eligible** for the loan at this time. Probability of eligibility: **{probability:.2%}**.")

        st.markdown("""
                #### Let’s Strengthen Your Application Together

                We believe in helping farmers like you build capacity and become financially ready. You can take advantage of our support programs to boost your eligibility:

                - **[Attend a Partner Training Center](https://yourdomain.com/training-centers)**  
                Learn about financial planning, record keeping, and digital tools that improve farm performance.

                - **[Join a Farmer Group](https://yourdomain.com/farmer-groups)**  
                Being part of a group increases trust and access to group-based loans and advisory services.

                - **[Visit a Farmer Education Hub](https://yourdomain.com/education-hubs)**  
                Get personalized mentorship, resources, and insights to strengthen your loan application.

                You're not far from being eligible — take these steps and reapply with confidence. We're here to support your journey! 🌱
                """)
