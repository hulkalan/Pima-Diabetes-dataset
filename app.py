from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    features = [float(x) for x in request.form.values()]

    # Scale and reshape input
    final_input = scaler.transform([features])

    # Make prediction
    prediction = model.predict(final_input)[0]

    # Result text
    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return render_template('index.html', prediction_text=f'Prediction: {result}')


if __name__ == '__main__':
    app.run(debug=True)
