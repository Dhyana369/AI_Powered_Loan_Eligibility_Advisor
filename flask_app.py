from flask import Flask, request, render_template
from flask import session, redirect, url_for
from markupsafe import escape 
import pickle
import numpy as np
import sqlite3
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

app = Flask(__name__)
app.secret_key = "your_secret_key"

model = pickle.load(open("E:\Project\AI_Powered_Loan_Eligibility_Advisor\model.pkl",'rb'))

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

        # One-hot logic
        male = 1 if gender == "Male" else 0
        married_yes = 1 if married == "Yes" else 0
        dependents_1 = 1 if dependents == '1' else 0
        dependents_2 = 1 if dependents == '2' else 0
        dependents_3 = 1 if dependents == '3+' else 0

        not_graduate = 1 if education == "Not Graduate" else 0
        employed_yes = 1 if employed == "Yes" else 0

        if area == "Semiurban":
            semiurban = 1
            urban = 0
        elif area == "Urban":
            semiurban = 0
            urban = 1
        else:
            semiurban = 0
            urban = 0

        ApplicantIncomeLog = np.log(ApplicantIncome + 1)
        totalincomelog = np.log(ApplicantIncome + CoapplicantIncome + 1)
        LoanAmountLog = np.log(LoanAmount + 1)
        Loan_Amount_Termlog = np.log(Loan_Amount_Term + 1)

        prediction = model.predict([[credit, ApplicantIncomeLog, LoanAmountLog,
                                     Loan_Amount_Termlog, totalincomelog, male,
                                     married_yes, dependents_1, dependents_2,
                                     dependents_3, not_graduate, employed_yes,
                                     semiurban, urban]])

        prediction_text = "Yes" if prediction == 1 else "No"

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

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
