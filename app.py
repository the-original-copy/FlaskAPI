from flask import Flask,request,jsonify
import numpy as np
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm

model = pickle.load(open('Diabetes_model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    t_glucose = request.form.get('glucose')
    t_bmi = request.form.get('BMI')
    t_dpf = request.form.get('DPF')
    t_age = request.form.get('age')
    glucose = float(t_glucose)
    bmi = float(t_bmi)
    dpf = float(t_dpf)
    age = float(t_age)
    input_query = np.array([[glucose,bmi,dpf,age,1]])
    predicted_probabilities = model.predict_proba(input_query)
    diabetic = predicted_probabilities[0,1]
    return jsonify({'Result':str(diabetic)})

if __name__ == '__main__':
    app.run(debug=True)

