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
    if request.is_json:
        data = request.get_json()
        features = [float(data[col]) for col in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]
    else:
        features = [float(x) for x in request.form.values()]

    final_input = scaler.transform([features])
    prediction = model.predict(final_input)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return {"prediction": result}


if __name__ == '__main__':
    app.run(debug=True)
