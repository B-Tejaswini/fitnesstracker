import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ✅ Set Page Configuration
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🏋", layout="wide")

# ✅ Function to set background image
def set_bg(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# ✅ Apply background (Ensure this image exists)
set_bg("fitness_tracker_background.png")

# ✅ Initialize session state for prediction
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ✅ Load Dataset
file_path = "fitness_tracker_dataset.csv"
df = None
try:
    df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df = df.dropna()
    st.write("📊 **Dataset Loaded Successfully!**")
except Exception as e:
    st.error(f"⚠ Error loading dataset: {e}")

# ✅ Sidebar Input Section
st.sidebar.header("📋 Enter Your Details")
age = st.sidebar.number_input("🎂 Age:", min_value=10, max_value=100, value=25, step=1)
weight = st.sidebar.number_input("⚖ Weight (kg):", min_value=30, max_value=200, value=70, step=1)
height = st.sidebar.number_input("📏 Height (cm):", min_value=100, max_value=250, value=170, step=1)
steps = st.sidebar.number_input("🚶 Daily Steps:", min_value=0, max_value=50000, value=5000, step=100)

# ✅ Exercise Timing Input
st.sidebar.header("⏳ Enter Exercise Duration (Minutes)")
exercise_duration = st.sidebar.number_input("⌛ Exercise Duration:", min_value=0, max_value=300, value=30, step=5)

# ✅ Calculate BMI & Health Status
bmi = weight / ((height / 100) ** 2)
health_status = "Underweight" if bmi < 18.5 else "Normal Weight" if bmi < 24.9 else "Overweight" if bmi < 29.9 else "Obese"
st.sidebar.write(f"🩺 **Health Status:** {health_status}")

# ✅ Train Model (if dataset is available)
if df is not None:
    # ✅ Check if 'Exercise_Duration' exists in dataset
    available_columns = ['Age', 'Weight_kg', 'Height_cm', 'Daily_Steps']
    if 'Exercise_Duration' in df.columns:
        available_columns.append('Exercise_Duration')

    try:
        X = df[available_columns]
        y = df['Calories_Burned']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        st.write("✅ **Model Trained Successfully!**")
    except KeyError as ke:
        st.error(f"⚠ Dataset issue: {ke}")

# ✅ Predict Calories Burned
if st.sidebar.button("🔥 Predict Calories Burned"):
    input_data = [[age, weight, height, steps, exercise_duration] if 'Exercise_Duration' in available_columns else [age, weight, height, steps]]
    if df is not None:
        st.session_state.prediction = model.predict(input_data)[0]

# ✅ Display Prediction Result
if st.session_state.prediction is not None:
    st.markdown(f"### 🔥 Estimated Calories Burned: *{round(st.session_state.prediction, 2)} kcal*")

# ✅ Save Data Button
if st.sidebar.button("💾 Save Data"):
    if st.session_state.prediction is not None:
        new_data = pd.DataFrame([[datetime.date.today(), age, weight, height, steps, exercise_duration, round(st.session_state.prediction, 2), round(bmi, 2)]],
                                columns=['Date', 'Age', 'Weight', 'Height', 'Steps', 'Exercise_Duration', 'Calories_Burned', 'BMI'])
        if os.path.exists("user_history.csv"):
            history = pd.read_csv("user_history.csv")
            history = pd.concat([history, new_data], ignore_index=True)
        else:
            history = new_data
        history.to_csv("user_history.csv", index=False)
        st.success("✅ Data saved successfully!")
    else:
        st.warning("⚠ Please predict calories burned before saving data!")

# ✅ Display Progress Over Time
if os.path.exists("user_history.csv"):
    history = pd.read_csv("user_history.csv")
    history["Date"] = pd.to_datetime(history["Date"], errors='coerce').dt.date  # ✅ Fix for date parsing issue
    fig = px.line(history, x="Date", y="Calories_Burned", title="📈 Calories Burned Over Time", markers=True)
    st.plotly_chart(fig)
