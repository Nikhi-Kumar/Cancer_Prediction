import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = r"C:\Users\boddu\Desktop\Cancer_Prediction\data\Cancer_Data_Modified.csv"
data = pd.read_csv(file_path)

# Drop rows with NaN values
data.dropna(inplace=True)

# Categorical columns to encode
label_cols = ['AlcoholIntake', 'DietType', 'StressLevels', 'OccupationType', 'BloodPressure']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and target
features = [
    'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity',
    'AlcoholIntake', 'CancerHistory', 'DietType', 'StressLevels',
    'SleepHours', 'RadiationExposure', 'OccupationType', 'BloodPressure',
    'DiabetesHistory'
]
X = data[features]
y = data['Diagnosis']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Gradient Boosting
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train, y_train)

# Ensemble
voting_clf = VotingClassifier(estimators=[('rf', best_rf), ('gbc', gbc)], voting='soft')
voting_clf.fit(X_train, y_train)

# Evaluation
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model, scaler, and encoders
joblib.dump(voting_clf, r"C:\Users\boddu\Desktop\Cancer_Prediction\cancer_prediction_model.pkl")
joblib.dump(scaler, r"C:\Users\boddu\Desktop\Cancer_Prediction\scaler.pkl")
joblib.dump(label_encoders, r"C:\Users\boddu\Desktop\Cancer_Prediction\label_encoders.pkl")
print("Model, scaler, and encoders saved successfully.")
