import os
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing objects
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')  # list of strings

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Use form input keys matching selected_features
        input_data = {name: float(request.form[name]) for name in selected_features}
        df = pd.DataFrame([input_data])

        scaled = scaler.transform(df)
        prediction = model.predict(scaled)
        result = 'Malignant' if prediction[0] == 1 else 'Benign'

        return render_template('index.html', prediction_text=f'Tumor Type: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Heroku fix
    app.run(debug=False, host='0.0.0.0', port=port)
