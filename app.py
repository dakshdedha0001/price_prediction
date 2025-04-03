import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 📌 **Title**
st.title("Onion Price Prediction using AI-ML")

# 📌 **Load Dataset**
@st.cache_data
def load_data():
    df = pd.read_csv("Price_Agriculture_commodities_Week.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()
st.write(df.head())

# 📌 **Data Visualization: Price Trend**
st.subheader("Onion Price Trend Over Time")
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Modal Price"], marker="o", linestyle="-", color="green")
plt.xlabel("Date")
plt.ylabel("Price (₹ per quintal)")
plt.title("Onion Price Trend")
st.pyplot(plt)

# 📌 **Data Preprocessing for LSTM**
df["Price"] = df["Modal Price"]
df["Price"] = df["Price"].fillna(method="ffill")  # Handle missing values

# Convert to supervised learning format
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 30
data = df["Price"].values
X, y = create_sequences(data, time_steps)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 📌 **LSTM Model**
st.subheader("Training LSTM Model")
model = Sequential([
    LSTM(50, activation="relu", return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(50, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

# 📌 **Predictions**
y_pred = model.predict(X_test)

# 📌 **Evaluation Metrics**
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"**Model RMSE:** {rmse:.2f}")
st.write(f"**Model R² Score:** {r2:.2f}")

# 📌 **Future Prediction**
st.subheader("Future Price Prediction")
future_prices = model.predict(X_test[-1].reshape(1, time_steps, 1))
st.write(f"**Predicted Next Day Price:** ₹{future_prices[0][0]:.2f}")

# 📌 **End of App**
st.write("🚀 Streamlit App Successfully Running!")
