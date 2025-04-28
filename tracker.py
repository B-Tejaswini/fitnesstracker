import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Streamlit UI
st.title("Personal Fitness Tracker 📊")
st.write("A Machine Learning-based Calories Burned Predictor")

# Load Dataset
file_path = r"C:\Users\tejut\Desktop\AICTE Internship\fitness_tracker_dataset.csv"  # Corrected file path

try:
    df = pd.read_csv(file_path, delimiter=",", encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.strip()  # Clean column names
    df = df.dropna()  # Remove missing values

    # Display Dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Check required columns exist
    required_columns = {'Age', 'Weight_kg', 'Height_cm', 'Daily_Steps', 'Calories_Burned'}
    if not required_columns.issubset(df.columns):
        st.error("Dataset is missing required columns. Please check the CSV file.")
    else:
        # Select Features & Target
        X = df[['Age', 'Weight_kg', 'Height_cm', 'Daily_Steps']]
        y = df['Calories_Burned']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model (Every Time the Script Runs)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        st.success("Model trained successfully! ✅")

        # User Input for Prediction
        st.subheader("Predict Calories Burned")
        age = st.number_input("Enter Age:", min_value=10, max_value=100, value=25)
        weight = st.number_input("Enter Weight (kg):", min_value=30, max_value=200, value=70)
        height = st.number_input("Enter Height (cm):", min_value=100, max_value=250, value=170)
        steps = st.number_input("Enter Daily Steps:", min_value=0, max_value=50000, value=5000)

        # Predict Button
        if st.button("Predict"):
            input_data = [[age, weight, height, steps]]
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Calories Burned: {round(prediction, 2)} kcal 🔥")

except FileNotFoundError:
    st.error(f"Error: The dataset file was not found at {file_path}. Please check the file path.")

except pd.errors.ParserError as e:
    st.error("Error: Issue while parsing the CSV file.")
    st.text(str(e))

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
