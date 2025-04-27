import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Title
st.title("Agri-Horticultural Commodities Price Prediction (Onion, Potato, etc.)")

# Load Data
data = pd.read_csv("Price_Agriculture_commodities_Week.csv")

# Select Commodity
commodity_list = data['Commodity'].unique()
selected_commodity = st.selectbox("Select a Commodity:", commodity_list)

# Filter based on selection
df = data[data['Commodity'] == selected_commodity]

# Date Features
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
df['Year'] = df['Arrival_Date'].dt.year
df['Month'] = df['Arrival_Date'].dt.month
df['Day'] = df['Arrival_Date'].dt.day

# Features and Target
features = ['Year', 'Month', 'Day', 'Min Price', 'Max Price']
target = 'Modal Price'
X = df[features]
y = df[target]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Clustering (KMeans)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Show Clusters
st.subheader("Cluster Analysis")
st.write(df[['Year', 'Month', 'Day', 'Min Price', 'Max Price', 'Cluster']])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=500)
mlp_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test)

# Evaluation
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
r2_mlp = r2_score(y_test, y_pred_mlp)

st.subheader("Model Evaluation Results")
st.write(f"XGBoost Model: RMSE = {rmse_xgb:.2f}, R² = {r2_xgb:.2f}")
st.write(f"MLP (ULNN) Model: RMSE = {rmse_mlp:.2f}, R² = {r2_mlp:.2f}")

# Visualization
st.subheader("Prediction Comparison")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_xgb, color='blue', alpha=0.5, label='XGBoost Predictions')
ax.scatter(y_test, y_pred_mlp, color='red', alpha=0.5, label='MLP Predictions')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Future Prediction
st.subheader("Predict Future Price")
year = st.number_input("Enter Year (2024-2030):", min_value=2024, max_value=2030, step=1)
month = st.number_input("Enter Month (1-12):", min_value=1, max_value=12, step=1)
day = st.number_input("Enter Day (1-31):", min_value=1, max_value=31, step=1)
min_price = st.number_input("Enter Min Price:")
max_price = st.number_input("Enter Max Price:")

if st.button("Predict"):
    input_data = scaler.transform([[year, month, day, min_price, max_price]])
    xgb_price = xgb_model.predict(input_data)[0]
    mlp_price = mlp_model.predict(input_data)[0]
    st.success(f"🔮 XGBoost Predicted Price: ₹{xgb_price:.2f}")
    st.success(f"🔮 MLP (ULNN) Predicted Price: ₹{mlp_price:.2f}")

