from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and encoders
model = joblib.load('cancer_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        bmi = float(request.form['bmi'])
        smoking = int(request.form['smoking'])
        genetic_risk = int(request.form['genetic_risk'])
        physical_activity = float(request.form['physical_activity'])
        alcohol_intake = request.form['alcohol_intake']
        family_history = int(request.form['family_history'])
        diet_type = request.form['diet_type']
        stress_levels = request.form['stress_levels']
        sleep_hours = float(request.form['sleep_hours'])
        radiation_exposure = int(request.form['radiation_exposure'])
        occupation_type = request.form['occupation_type']
        blood_pressure = request.form['blood_pressure']
        diabetes_history = int(request.form['diabetes_history'])

        # Encode categorical variables
        alcohol_intake_encoded = label_encoders['AlcoholIntake'].transform([alcohol_intake])[0]
        diet_type_encoded = label_encoders['DietType'].transform([diet_type])[0]
        stress_levels_encoded = label_encoders['StressLevels'].transform([stress_levels])[0]
        occupation_type_encoded = label_encoders['OccupationType'].transform([occupation_type])[0]
        blood_pressure_encoded = label_encoders['BloodPressure'].transform([blood_pressure])[0]

        # Prepare input
        features = np.array([[
            age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake_encoded,
            family_history, diet_type_encoded, stress_levels_encoded, sleep_hours, radiation_exposure,
            occupation_type_encoded, blood_pressure_encoded, diabetes_history
        ]])

        # Standardize input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        # Determine risk level for background color
        if prediction == 1:
            risk_level = "high"
        elif prediction == 0:
            risk_level = "low"
        else:
            risk_level = "medium"

        return render_template('result.html', prediction=prediction, risk_level=risk_level)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
