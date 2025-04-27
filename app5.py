# 📌 Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# 📌 Step 1: Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Price_Agriculture_commodities_Week.csv")
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
    df["Year"] = df["Arrival_Date"].dt.year
    df["Month"] = df["Arrival_Date"].dt.month
    df["Day"] = df["Arrival_Date"].dt.day
    return df

# 📌 Step 2: Train models for each commodity
@st.cache_resource
def train_models(df):
    features = ["Year", "Month", "Day", "Min Price", "Max Price"]
    target = "Modal Price"

    commodity_models = {}
    commodity_scalers = {}

    for commodity in df["Commodity"].unique():
        df_commodity = df[df["Commodity"] == commodity]
        X = df_commodity[features]
        y = df_commodity[target]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        commodity_models[commodity] = model
        commodity_scalers[commodity] = scaler

    return commodity_models, commodity_scalers

# 📌 Main App
def main():
    st.title("🌾 Agri-Horticultural Commodity Price Prediction")
    st.subheader("Developed using ULNN (MLP Neural Network)")

    df = load_data()
    commodity_models, commodity_scalers = train_models(df)

    st.sidebar.header("Select Inputs")

    commodities = sorted(df["Commodity"].unique())
    selected_commodity = st.sidebar.selectbox("Select Commodity", commodities)

    year = st.sidebar.number_input("Enter Year (2024-2030)", min_value=2024, max_value=2030, value=2024)
    month = st.sidebar.number_input("Enter Month (1-12)", min_value=1, max_value=12, value=1)
    day = st.sidebar.number_input("Enter Day (1-31)", min_value=1, max_value=31, value=1)
    min_price = st.sidebar.number_input("Enter Estimated Min Price", min_value=0.0, value=10.0)
    max_price = st.sidebar.number_input("Enter Estimated Max Price", min_value=0.0, value=20.0)

    if st.sidebar.button("Predict Price"):
        input_data = np.array([[year, month, day, min_price, max_price]])
        scaler = commodity_scalers[selected_commodity]
        input_scaled = scaler.transform(input_data)

        model = commodity_models[selected_commodity]
        predicted_price = model.predict(input_scaled)[0]

        st.success(f"🔮 Predicted Modal Price for **{selected_commodity}**: ₹{predicted_price:.2f}")

        # 📈 Visualization
        st.subheader("Model Performance Visualization")
        df_commodity = df[df["Commodity"] == selected_commodity]
        X = df_commodity[["Year", "Month", "Day", "Min Price", "Max Price"]]
        y = df_commodity["Modal Price"]
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        plt.figure(figsize=(8, 5))
        plt.scatter(y, y_pred, color='green', alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel("Actual Modal Price")
        plt.ylabel("Predicted Modal Price")
        plt.title(f"Prediction Performance: {selected_commodity}")
        plt.grid(True)
        st.pyplot(plt)

    st.write("---")
    st.write("📚 **Project:** Development of AI-ML based models for predicting prices of agri-horticultural commodities such as pulses and vegetables (onion, potato, etc.)")
    st.write("👨‍💻 **Developed Using:** Unsupervised Learning + Neural Networks")

if __name__ == "__main__":
    main()
