import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model components
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and align input with selected features
        input_data = {}
        for feature in selected_features:
            value = float(request.form.get(feature, 0))
            input_data[feature] = value

        df = pd.DataFrame([input_data], columns=selected_features)

        # Scale and predict
        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)
        result = 'Malignant' if prediction[0] == 1 else 'Benign'

        return render_template('index.html', prediction_text=f'Tumor Type: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
