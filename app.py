import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and selected features
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')  # List of feature names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values in correct feature order
        input_data = {feature: float(request.form[feature]) for feature in selected_features}
        df = pd.DataFrame([input_data], columns=selected_features)

        # Scale and predict
        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)
        result = 'Malignant' if prediction[0] == 1 else 'Benign'

        return render_template('index.html', prediction_text=f'Tumor Type: {result}')
    except Exception as e:
        # Show detailed error in UI for debugging
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
