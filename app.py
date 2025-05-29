from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing objects
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')  # List of feature names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input into a dict using feature names
        input_data = {name: float(request.form[name]) for name in selected_features}
        df = pd.DataFrame([input_data])  # Create DataFrame with column names

        # Scale and predict
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)
        result = 'Malignant' if prediction[0] == 1 else 'Benign'

        return render_template('index.html', prediction_text=f'Tumor Type: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
