import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form input
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        return render_template('index.html', prediction_text=f'Tumor Type: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Heroku's dynamic port
    app.run(debug=True, host='0.0.0.0', port=port)
