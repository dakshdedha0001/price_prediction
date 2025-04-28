import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("Price_Agriculture_commodities_Week.csv")

# Data Preprocessing
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day

# Feature and Target
features = ["Year", "Month", "Day", "Min Price", "Max Price"]
target = "Modal Price"

X = df[features]
y = df[target]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Neural Network Model (ULNN Alternative)
model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=500)
model.fit(X_train, y_train)

# Streamlit App
st.title("🧅 Agri-Horticultural Commodity Price Prediction App")

# Select Commodity
commodity_options = df["Commodity"].unique()
selected_commodity = st.selectbox("Select Commodity", commodity_options)

# Input Features
st.subheader("Enter the Details:")
year = st.number_input("Year", min_value=2024, max_value=2030, value=2024)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
min_price = st.number_input("Minimum Price", min_value=0)
max_price = st.number_input("Maximum Price", min_value=0)

# Predict Button
if st.button("Predict Price"):
    input_data = scaler.transform([[year, month, day, min_price, max_price]])
    prediction = model.predict(input_data)[0]
    st.success(f"🔮 Predicted Modal Price: ₹{prediction:.2f}")
