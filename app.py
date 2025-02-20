import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Glucose Guardian AI")
st.write("Welcome to the diabetes prediction app!")

# Adding dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree_function", "age", "outcome"]
data = pd.read_csv(url, names=column_names)

# Separate features and target
X = data.drop("outcome", axis=1)
y = data["outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and model training
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Function to predict diabetes for a new patient
def predict_diabetes(patient_data):
    patient_df = pd.DataFrame([patient_data], columns=X.columns)
    prediction = pipeline.predict(patient_df)
    probability = pipeline.predict_proba(patient_df)[0][1]
    return prediction[0], probability

# Streamlit UI
st.write("This machine learning-powered app predicts diabetes risk based on blood glucose levels and other health parameters.")

# User input fields
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=6)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=148)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=72)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=35)
insulin = st.number_input('Insulin', min_value=0, max_value=800, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=33.6)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.627)
age = st.number_input('Age', min_value=18, max_value=100, value=50)

# Prepare the input data for prediction
new_patient_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

# Get prediction and probability
if st.button('Predict Diabetes'):
    prediction, probability = predict_diabetes(new_patient_data)

    # Display results
    st.write(f"### Prediction: {'🟢 Negative' if prediction == 0 else '🔴 Positive'}")
    st.write(f"#### Probability of diabetes: {probability:.2f}")

    # Feature Importance Visualization
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": pipeline.named_steps['classifier'].feature_importances_
    }).sort_values("importance", ascending=False)

    st.write("### Feature Importance:")
    st.bar_chart(feature_importance.set_index("feature"))

    # Display Correlation Heatmap
    st.write("### Correlation Heatmap of Features:")
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt.gcf())  # Fix: Ensuring correct figure is passed

# Additional Visualization: Glucose Levels & Diabetes Risk
st.write("### Impact of Glucose Levels on Diabetes Risk:")
plt.figure(figsize=(10, 6))
sns.boxplot(x='outcome', y='glucose', data=data)
plt.title('Impact of Glucose Levels on Diabetes Risk')
plt.xlabel('Diabetes Outcome (0: Negative, 1: Positive)')
plt.ylabel('Glucose Level')
st.pyplot(plt.gcf())  # Fix: Ensuring correct figure is passed
