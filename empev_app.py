import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_excel('employee_performance.xlsx')
    df = df.drop(['EmpNumber','Attrition'], axis=1)  # Dropping unnecessary columns
    
    # Encoding categorical variables using OrdinalEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    
    # Define features and target variable
    X = df.drop('PerformanceRating', axis=1)
    y = df['PerformanceRating']
    
    # Adjusting the target variable to start from 0
    y = y - y.min()
    
    # Balance the dataset using RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)
    
    return X, y, encoder, categorical_cols

# Train the XGBoost model with GridSearchCV
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the XGBoost model and set up the parameter grid for tuning
    xgb_model = XGBClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],  # Adjusted for lighter load
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return best_model, accuracy, report, X_train.columns

# Streamlit app
st.title("Employee Performance Prediction with Tuned XGBoost")

st.write("""
This app uses a tuned XGBoost model with GridSearchCV to predict employee performance based on historical data.
You can use it to assess the potential performance of new employees.
""")

# Input for Employee Name at the start of the form
employee_name = st.text_input("Employee Name", "John Doe")

# Load data and train model
X, y, encoder, categorical_cols = load_data()
model, accuracy, report, feature_order = train_model(X, y)

st.write(f"Model Accuracy: {accuracy}")

st.subheader("Predict Employee Performance")

# Define unique options for categorical variables
gender_options = ["Male", "Female"]
marital_status_options = ["Single", "Married", "Divorced"]
education_background_options = ["Life Sciences", "Marketing", "Human Resources", "Technical Degree", "Medical", "Other"]
department_options = ["Sales", "Research & Development", "Human Resources"]
job_role_options = ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Research Director", "Human Resources", "Sales Representative"]
business_travel_options = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
overtime_options = ["Yes", "No"]

# Collect user input for new employee data
input_data = {
    "Gender": st.selectbox("Gender", gender_options),
    "MaritalStatus": st.selectbox("Marital Status", marital_status_options),
    "EducationBackground": st.selectbox("Education Background", education_background_options),
    "EmpDepartment": st.selectbox("Department", department_options),
    "EmpJobRole": st.selectbox("Job Role", job_role_options),
    "BusinessTravelFrequency": st.selectbox("Business Travel Frequency", business_travel_options),
    "OverTime": st.selectbox("Overtime", overtime_options),
    "Age": st.number_input("Age", min_value=18, max_value=60, value=30),
    "DistanceFromHome": st.number_input("Distance From Home", min_value=1, max_value=29, value=10),
    "EmpEducationLevel": st.number_input("Education Level", min_value=1, max_value=5, value=3),
    "EmpEnvironmentSatisfaction": st.number_input("Environment Satisfaction", min_value=1, max_value=4, value=3),
    "EmpJobInvolvement": st.number_input("Job Involvement", min_value=1, max_value=4, value=3),
    "EmpJobLevel": st.number_input("Job Level", min_value=1, max_value=5, value=2),
    "EmpJobSatisfaction": st.number_input("Job Satisfaction", min_value=1, max_value=4, value=3),
    "NumCompaniesWorked": st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2),
    "EmpLastSalaryHikePercent": st.number_input("Last Salary Hike Percent", min_value=11, max_value=25, value=15),
    "EmpRelationshipSatisfaction": st.number_input("Relationship Satisfaction", min_value=1, max_value=4, value=3),
    "TotalWorkExperienceInYears": st.number_input("Total Work Experience in Years", min_value=0, max_value=40, value=10),
    "TrainingTimesLastYear": st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2),
    "EmpWorkLifeBalance": st.number_input("Work-Life Balance", min_value=1, max_value=4, value=3),
    "ExperienceYearsAtThisCompany": st.number_input("Experience Years at This Company", min_value=0, max_value=40, value=7),
    "ExperienceYearsInCurrentRole": st.number_input("Experience Years in Current Role", min_value=0, max_value=18, value=5),
    "YearsSinceLastPromotion": st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1),
    "YearsWithCurrManager": st.number_input("Years With Current Manager", min_value=0, max_value=17, value=3),
    "EmpHourlyRate": st.number_input("Hourly Rate", min_value=0, max_value=100, value=20),
}

# Convert categorical inputs using the same encoder used during training
input_df = pd.DataFrame([input_data])
input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])

# Ensure the input features are in the correct order
input_df = input_df[feature_order]

# Predict performance and show prediction probabilities
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.write(f"Employee Name: {employee_name}")
st.write(f"Predicted Performance Rating: {prediction[0]}")
st.write("Prediction Probabilities:")
st.write(prediction_proba)


        
st.write("Developed for INX Future Inc.")
st.write("Copyright © 2024. All rights reserved.")

