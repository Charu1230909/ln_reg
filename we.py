import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Weather Prediction", layout="centered")

st.title("🌦️ Weather Temperature Prediction using Linear Regression")

# Upload CSV
uploaded_file = st.file_uploader("Upload weather_data.csv", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Check required columns
    required_cols = {"humidity_level", "hours_sunlight", "daily_temperature"}
    if not required_cols.issubset(data.columns):
        st.error(
            "CSV must contain columns: humidity_level, hours_sunlight, daily_temperature"
        )
        st.stop()

    # Features and target
    X = data[["humidity_level", "hours_sunlight"]]
    y = data["daily_temperature"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("📊 Model Evaluation")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R² Score:", r2_score(y_test, y_pred))

    # User input
    st.subheader("🔍 Predict Temperature")
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
    hours_sunlight = st.number_input(
        "Hours of Sunlight", min_value=0.0, max_value=15.0, value=6.0
    )

    if st.button("Predict Temperature"):
        predicted_temp = model.predict([[humidity, hours_sunlight]])
        st.success(
            f"Predicted Temperature: {predicted_temp[0]:.2f} °C"
        )

    # Visualization
    st.subheader("📈 Visualization")
    fig, ax = plt.subplots()
    ax.scatter(data["humidity_level"], data["daily_temperature"])
    ax.set_xlabel("Humidity (%)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Humidity vs Temperature")

    st.pyplot(fig)

else:
    st.info("👆 Please upload the weather CSV file to begin.")
