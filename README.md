# **AI Powered Loan Eligibility Advisor**

A complete end-to-end Machine Learning project that predicts **loan eligibility** with an interactive UI.
This project was developed as part of my **Infosys Springboard Internship**, focusing on real-world ML application development and deployment.

---

## 🧠 **Overview**

The Loan Eligibility Advisor analyzes user-provided financial information and predicts whether an applicant is likely to be eligible for a loan.

It includes:

* A **Streamlit** web application (primary interface)
* A **Flask** backend (secondary interface with login system)
* A trained **ML model** built using scikit-learn
* A **chatbot** assistant powered by Google Generative AI
* SQLite-based user authentication (Flask version)

---

## 🚀 **Features**

✔️ Loan eligibility prediction using trained ML model

✔️ Clean and user-friendly Streamlit UI

✔️ Flask login system (optional secondary app)

✔️ ML preprocessing using scaler + model pipelines

✔️ EDA & model training notebook included

✔️ Modular, easy-to-understand project structure

---

## 📂 **Project Structure**

```
├── static/                     # Assets for Flask app
├── templates/                  # HTML templates for Flask
├── Loan_Model_Training.ipynb   # Model training notebook
├── flask_app.py                # Secondary backend with login
├── streamlit_app.py            # Primary app
├── chatbot.py                  # Chatbot logic
├── database.py                 # SQLite user management
├── model.pkl                   # Trained ML model
├── scaler.pkl                  # Saved scaler object
├── train.csv                   # Training dataset
├── system_architecture.png     # Diagram (optional)
├── users.db                    # SQLite database
└── README.md
```

---

## 🛠️ **Technologies Used**

### **Machine Learning**

* scikit-learn
* pandas
* numpy

### **Frontend / UI**

* Streamlit (primary)
* Flask with HTML templates (secondary)

### **AI Assistant**

* Google Generative AI (`google-generativeai`)

### **Others**

* SQLite
* matplotlib & seaborn (used in EDA)
* joblib / pickle for model persistence

---

## 📦 **Installation & Setup**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Dhyana369/AI_Powered_Loan_Eligibility_Advisor.git
cd AI_Powered_Loan_Eligibility_Advisor
```

### 2️⃣ Create and activate virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install required packages

```bash
pip install -r requirements.txt
```

---

## ▶️ **Run the Application**

### **Run Streamlit (Primary App)**

```bash
streamlit run streamlit_app.py
```

### **Run Flask App (Secondary Optional App)**

```bash
python flask_app.py
```

---

## 🎯 **How It Works**

1. User enters loan-related information
2. App preprocesses inputs using the saved scaler
3. Trained ML model predicts eligibility

---

## 💡 **Internship Note — Infosys Springboard**

This project was created during my **Infosys Springboard Internship**, where I worked on building a realistic ML application pipeline, including training, deployment, backend integration, authentication, and frontend UI development.

The experience strengthened my understanding of:

* End-to-end Data Science workflows
* Production-style ML deployment
* Streamlit and Flask integration
* Working with databases
* Communicating model insights effectively

---

## 🤝 **Contributing**

Contributions are welcome!
Feel free to open issues, submit pull requests, or suggest improvements.

---

## 📄 **License**

This project is licensed under the **MIT License**.
See the `LICENSE.txt` file for details.

---

## 📬 **Contact**

**Author:** Dhyana Anbalagan

**GitHub:** [https://github.com/Dhyana369](https://github.com/Dhyana369)

**Email:** *dhyanaanbalagan@gmail.com*

**LinkedIn:** *www.linkedin.com/in/dhyana-anbalagan*
