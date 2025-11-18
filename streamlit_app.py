import streamlit as st
import pickle
import numpy as np
import pandas as pd
from chatbot import show_chatbot

# Load the pre-trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the exact feature order expected by the model
TRAIN_COLUMNS = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                 'Credit_History', 'Property_Area']

# Streamlit Layout for Home Page
def home_page():
    st.title("Loan Prediction System")
    st.markdown("### 📋 **Welcome to the Loan Prediction System!**")
    st.markdown("""This tool helps you predict whether your loan application will be approved or rejected based on a variety of personal and financial factors. Fill in your details and let the system predict your loan status with an accurate machine learning model.""")

    # Add an image to the home page
    st.image("loan.png", caption="Loan Prediction System")

    st.markdown("### 🛠️ Project Overview")
    st.markdown("""
        This project was developed as part of my **AI internship** at **Infosys Springboard**. The objective was to build a **Loan Prediction System** that uses machine learning to predict whether a loan application will be approved or rejected based on various parameters like personal and financial details of the applicant.
        The system takes the user's input and processes it through a pre-trained model to deliver a prediction about the loan status. We used algorithms like **Logistic Regression** and **Decision Trees** for accurate predictions.
    """)

# Streamlit Layout for About Us Page
def about_us_page():
    st.title("📖 About Us")
    st.markdown("""**Our Mission**: We aim to provide **innovative, efficient, and easy-to-use** financial tools that 
                assist individuals in making better financial decisions. Our mission is to simplify 
                complex loan application processes using advanced technology like **Machine Learning** and **AI**.""")

    # System Architecture Section
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
        The **Loan Prediction System** architecture is designed to seamlessly collect user inputs, process them through a machine learning model, and deliver accurate predictions for loan approval. Here's a breakdown of the system components:
        
        1. **User Interface (UI)**: The front-end interface built using **Streamlit** allows users to input their personal and financial information.
        2. **Data Preprocessing**: Data entered by users is cleaned and transformed into numerical values, ensuring compatibility with the machine learning model.
        3. **Machine Learning Model**: The core of the system is a trained model (e.g., Random Forest, Logistic Regression), which predicts the likelihood of loan approval based on historical data.
        4. **Prediction Engine**: The system runs the model on the preprocessed data and returns the loan approval decision.
        5. **Visualization & Output**: The result is displayed to the user in a clear format with feedback on whether the loan is likely to be approved or rejected.
        
        Below is a diagram of the system architecture for better understanding:
    """)

    # Image of System Architecture
    st.image("system_architecture.png", caption="System Architecture Diagram")

    # Activity Log Section
    st.markdown("### 📊 Project Activity Log")
    st.markdown("""
        Throughout my internship, I engaged in various key activities to contribute to the **Loan Prediction System** project:
        
        1. **Data Collection**: Gathered relevant data from past loan applicants, including financial history, loan amounts, and approval outcomes.
        2. **Data Cleaning**: Handled missing values, outliers, and transformed categorical data to ensure compatibility with machine learning algorithms.
        3. **Model Selection**: Experimented with various machine learning algorithms such as **Logistic Regression**, **Random Forest**, and **XGBoost** to find the best-performing model.
        4. **Model Training & Optimization**: Trained the model on historical data, fine-tuned hyperparameters, and evaluated performance metrics like **accuracy**, **precision**, and **recall**.
        5. **Deployment & Testing**: Deployed 
        the trained model into the Streamlit application, tested the prediction system with real user inputs, and optimized its performance.

        This project has provided me with hands-on experience in **data preprocessing**, **model training**, and **AI deployment**, which are essential skills for any AI professional.
    """)

    # Project Activity Image
    st.image("Project Activity.png", caption="Project Activity Diagram")

# Streamlit Layout for Prediction Page
def prediction_page():
    st.title("🚀 Loan Prediction System")
    st.markdown("### 📋 Enter Loan Application Details to Predict Your Loan Status!")

    # User input fields
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    married = st.selectbox("💍 Marital Status", ["Yes", "No"])
    dependents = st.selectbox("👨‍👩‍👧 Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
    employed = st.selectbox("💼 Self Employed", ["Yes", "No"])
    credit = st.slider("📊 Credit score", min_value=300, max_value=850, step=1, value=750)
    area = st.selectbox("🏠 Property Area", ["Urban", "Semiurban", "Rural"])
    ApplicantIncome = st.slider("💰 Applicant Income", min_value=1000, max_value=100000, step=1000, value=5000)
    CoapplicantIncome = st.slider("🤝 Coapplicant Income", min_value=0, max_value=100000, step=1000, value=0)
    LoanAmount = st.slider("🏦 Loan Amount", min_value=1, max_value=10000, step=10, value=150)
    st.info("💡 **Note:** For best results, use loan amounts between $50-$700 (the model was trained on this range). Larger amounts may be rejected.")
    Loan_Amount_Term = st.select_slider("📅 Loan Amount Term (in days)", options=[360, 180, 240, 120], value=360)

        # Preprocess input for the trained model (exactly 11 features)
    def preprocess_data(gender, married, dependents, education, employed, credit, area,
                        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
        """
        Preprocess user inputs to match the model's expected format.
        Uses LabelEncoder-style encoding as used during training.
        """
        # Gender: Male=1, Female=0 (LabelEncoder typically encodes alphabetically: Female=0, Male=1)
        gender_encoded = 1 if gender == "Male" else 0
        
        # Married: Yes=1, No=0
        married_encoded = 1 if married == "Yes" else 0
        
        # Dependents: Convert "3+" to 3, otherwise use the number
        dependents_encoded = 3 if dependents == "3+" else int(dependents)
        
        # Education: Graduate=0, Not Graduate=1 (LabelEncoder encodes alphabetically: Graduate=0, Not Graduate=1)
        education_encoded = 0 if education == "Graduate" else 1
        
        # Self_Employed: Yes=1, No=0
        self_employed_encoded = 1 if employed == "Yes" else 0
        
        # Property_Area: Rural=0, Semiurban=1, Urban=2 (LabelEncoder encodes alphabetically)
        if area == "Rural":
            property_area_encoded = 0
        elif area == "Semiurban":
            property_area_encoded = 1
        else:  # Urban
            property_area_encoded = 2
        
        # Credit_History: Convert credit score (300-850) to binary (0 or 1)
        # Typically: credit score >= 650-700 is considered good (1), else (0)
        # Using 700 as threshold for good credit history
        credit_history_encoded = 1 if credit >= 700 else 0

        # Return features in the exact order the model expects (matching TRAIN_COLUMNS)
        features = [
            gender_encoded,           # Gender
            married_encoded,          # Married
            dependents_encoded,       # Dependents
            education_encoded,        # Education
            self_employed_encoded,    # Self_Employed
            ApplicantIncome,          # ApplicantIncome
            CoapplicantIncome,        # CoapplicantIncome
            LoanAmount,               # LoanAmount
            Loan_Amount_Term,         # Loan_Amount_Term
            credit_history_encoded,   # Credit_History
            property_area_encoded     # Property_Area
        ]
        
        return features

    if st.button("🔮 Predict Loan Status"):
        # Preprocess inputs
        features = preprocess_data(
            gender, married, dependents, education, employed, credit, area,
            ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
        )

        # Convert to DataFrame with correct feature names and order
        input_df = pd.DataFrame([features], columns=TRAIN_COLUMNS)
        
        # Scale the input using the same scaler used during training
        input_scaled = scaler.transform(input_df)

        # Make prediction (model returns 0 for 'N' or 1 for 'Y')
        prediction = model.predict(input_scaled)[0]

        # Display prediction with messages
        # Model returns: 0 = 'N' (Rejected), 1 = 'Y' (Approved)
        if prediction == 0:
            st.error("⚠️ Loan Status: **Rejected**")
            st.markdown("### ☠️ Danger! Your loan application has been **rejected**.")
            st.markdown("""
            **Possible reasons:**
            - Low credit history score
            - Insufficient income for the loan amount requested
            - High debt-to-income ratio
            - Other potential risk factors

            **Critical suggestions:**
            - **Immediate action:** Improve your credit score by paying off existing debts.
            - Consider **reducing your loan amount** or opting for a longer repayment term.
            - **Reassess your finances** and improve your overall financial health before reapplying.
            """)
        else:  # prediction == 1
            st.success("✅ Loan Status: **Approved**")
            st.markdown("### 🎉 Congratulations! Your loan application is likely approved!")
            st.balloons()
            st.markdown("""
            **Key highlights of your application:**
            - Good credit history score
            - Sufficient income to cover loan repayment
            - Positive factors supporting your loan approval
            """)


def show_chatbot_page():
    # Link the chatbot to the chatbot page
    show_chatbot()

# Footer for About Us Page
def footer():
    st.markdown("---")
    st.markdown("### 🌐 Connect with us")
    st.markdown("LinkedIn: https://www.linkedin.com/in/dhyana-anbalagan")
    st.markdown("GitHub: https://github.com/Dhyana369")
    st.markdown("📧 Email: dhyanaanbalagan@gmail.com")

# Sidebar Layout Design Enhancement
def sidebar_layout():
    st.sidebar.title("🔧 Menu")
    st.sidebar.markdown("### Choose a Page")
    
    menu = st.sidebar.radio(
        "Go to", ["Home", "About Us", "Prediction", "Chatbot"]
    )
    
    if menu == "Home":
        home_page()
    elif menu == "Prediction":
        prediction_page()
    elif menu == "Chatbot":
        show_chatbot_page()
    else:
        about_us_page()
        footer()

# Set initial page state if not defined
sidebar_layout()