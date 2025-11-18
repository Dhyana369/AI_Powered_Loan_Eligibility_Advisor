from flask import Flask, request, render_template, jsonify
from flask import session, redirect, url_for
from markupsafe import escape 
import pickle
import numpy as np
import pandas as pd
import sqlite3
import os

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

app = Flask(__name__)
app.secret_key = os.getenv("GEMINI_API_KEY")

MODEL_PATH = r"E:\Project\AI_Powered_Loan_Eligibility_Advisor\model.pkl"
SCALER_PATH = r"E:\Project\AI_Powered_Loan_Eligibility_Advisor\scaler.pkl"

model = pickle.load(open(MODEL_PATH, 'rb'))
scaler = pickle.load(open(SCALER_PATH, 'rb'))

TRAIN_COLUMNS = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area'
]

CHAT_QUESTIONS = [
    ("gender", "What is your gender? (Male/Female)"),
    ("married", "Are you married? (Yes/No)"),
    ("dependents", "How many dependents? (0/1/2/3+)"),
    ("education", "What is your education level? (Graduate/Not Graduate)"),
    ("self_employed", "Are you self-employed? (Yes/No)"),
    ("applicant_income", "What is your monthly applicant income?"),
    ("coapplicant_income", "What is your monthly co-applicant income?"),
    ("loan_amount", "What loan amount are you requesting?"),
    ("loan_amount_term", "Loan term (in days)?"),
    ("credit_history", "What is your credit score? (300-850)"),
    ("property_area", "Which property area? (Urban/Semiurban/Rural)")
]


def encode_features(gender, married, dependents, education, employed, credit,
                    area, applicant_income, coapplicant_income,
                    loan_amount, loan_term):
    gender_encoded = 1 if str(gender).strip().lower() == "male" else 0
    married_encoded = 1 if str(married).strip().lower() == "yes" else 0
    dependents_encoded = 3 if str(dependents).strip() == "3+" else int(float(dependents))
    education_encoded = 0 if str(education).strip().lower() == "graduate" else 1
    self_employed_encoded = 1 if str(employed).strip().lower() == "yes" else 0

    area_lower = str(area).strip().lower()
    if area_lower == "rural":
        property_area_encoded = 0
    elif area_lower == "semiurban":
        property_area_encoded = 1
    else:
        property_area_encoded = 2

    credit_history_encoded = 1 if float(credit) >= 700 else 0

    return [
        gender_encoded,
        married_encoded,
        dependents_encoded,
        education_encoded,
        self_employed_encoded,
        float(applicant_income),
        float(coapplicant_income),
        float(loan_amount),
        float(loan_term),
        credit_history_encoded,
        property_area_encoded
    ]


def init_chat_state():
    state = session.get('chat_state')
    if not state:
        state = {"started": False, "current_step": -1, "responses": {}}
    session['chat_state'] = state
    return state


def reset_chat_state():
    session['chat_state'] = {"started": False, "current_step": -1, "responses": {}}


def validate_chat_response(step_key, message):
    msg = message.strip()
    lower = msg.lower()

    if step_key == "gender":
        if lower in ("male", "female"):
            return True, msg.title()
        return False, "Please answer with Male or Female."

    if step_key == "married":
        if lower in ("yes", "no"):
            return True, lower.capitalize()
        return False, "Answer Yes or No."

    if step_key == "dependents":
        if lower in ("0", "1", "2", "3+"):
            return True, "3+" if lower == "3+" else str(int(lower))
        return False, "Dependents must be 0, 1, 2 or 3+."

    if step_key == "education":
        if lower in ("graduate", "not graduate"):
            return True, "Graduate" if lower == "graduate" else "Not Graduate"
        return False, "Please answer Graduate or Not Graduate."

    if step_key == "self_employed":
        if lower in ("yes", "no"):
            return True, lower.capitalize()
        return False, "Answer Yes or No."

    if step_key in ("applicant_income", "coapplicant_income", "loan_amount", "loan_amount_term"):
        try:
            value = float(msg)
            if value <= 0:
                return False, "Value must be positive."
            return True, value
        except ValueError:
            return False, "Please enter a numeric value."

    if step_key == "credit_history":
        try:
            score = float(msg)
            if 0 <= score <= 1000:
                return True, score
            return False, "Credit score must be between 0 and 1000."
        except ValueError:
            return False, "Please enter a numeric credit score."

    if step_key == "property_area":
        if lower in ("urban", "semiurban", "rural"):
            return True, msg.title()
        return False, "Answer Urban, Semiurban, or Rural."

    return True, msg

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=["GET","POST"])
def predict():

    # optional protection
    # if 'user' not in session:
    #     return redirect(url_for('login'))

    if request.method == 'POST':
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit  = float(request.form['credit'])
        area = request.form['area']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])

        features = encode_features(
            gender,
            married,
            dependents,
            education,
            employed,
            credit,
            area,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term
        )

        input_df = pd.DataFrame([features], columns=TRAIN_COLUMNS)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        prediction_text = "Approved" if prediction == 1 else "Rejected"

        return render_template("prediction.html", 
                               prediction_text=f"Loan status is {prediction_text}")

    return render_template("prediction.html")


import sqlite3

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cur.fetchone()
        conn.close()

        if user:
            session['username'] = username     # <-- THIS IS THE IMPORTANT LINE
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()

            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()

            return redirect(url_for('login'))

        except:
            return render_template('signup.html', message="Username already exists!")

    return render_template('signup.html')


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat_api', methods=["POST"])
def chat_api():
    state = init_chat_state()
    data = request.get_json()
    message = data.get("message", "").strip()

    # If user says start/restart
    if message.lower() in ("start", "yes", "restart", "begin"):
        reset_chat_state()
        state = init_chat_state()
        state["started"] = True
        state["current_step"] = 0
        session["chat_state"] = state
        session.modified = True

        return jsonify({
            "reply": CHAT_QUESTIONS[0][1]   # first question text
        })

    # If not started, ask to start
    if not state["started"]:
        return jsonify({
            "reply": "Type 'start' to begin the loan eligibility check."
        })

    step = state["current_step"]
    step_key, question_text = CHAT_QUESTIONS[step]

    # Validate the user response
    valid, result = validate_chat_response(step_key, message)
    if not valid:
        return jsonify({"reply": result})  # return validation error message

    # Store valid response
    state["responses"][step_key] = result

    # Move to next step
    state["current_step"] += 1
    session["chat_state"] = state
    session.modified = True

    # If all questions answered → Make prediction
    if state["current_step"] >= len(CHAT_QUESTIONS):
        r = state["responses"]

        features = encode_features(
            r["gender"],
            r["married"],
            r["dependents"],
            r["education"],
            r["self_employed"],
            r["credit_history"],
            r["property_area"],
            r["applicant_income"],
            r["coapplicant_income"],
            r["loan_amount"],
            r["loan_amount_term"]
        )

        input_df = pd.DataFrame([features], columns=TRAIN_COLUMNS)
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        result_text = "Approved" if pred == 1 else "Rejected"

        reset_chat_state()
        session.modified = True

        return jsonify({
            "reply": "Thanks! Your loan eligibility result is:",
            "prediction": result_text
        })

    # Ask next question
    return jsonify({"reply": CHAT_QUESTIONS[step][1]})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
