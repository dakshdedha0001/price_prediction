import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# **Step 1: Load Data**
@st.cache_data
def load_data():
    df = pd.read_csv("Price_Agriculture_commodities_Week.csv")  # Apna file path check karna
    df = df[df["Commodity"] == "Onion"]  # Sirf Onion ka data filter karo
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
    return df

df = load_data()

# **Step 2: Streamlit UI**
st.title("📊 Onion Price Prediction App 🚀")
st.write("Agri-Horticulture Commodity Price Prediction Using AI-ML")

# **Step 3: Data Exploration**
st.subheader("🔍 Data Preview")
st.write(df.head())

# **Step 4: Data Preprocessing**
st.subheader("📦 Preparing Data for Model")
df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day

features = ["Year", "Month", "Day", "Min Price", "Max Price"]
target = "Modal Price"

X = df[features]
y = df[target]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# **Step 5: Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# **Step 6: Train Model**
model = LinearRegression()
model.fit(X_train, y_train)

# **Step 7: Make Predictions**
y_pred = model.predict(X_test)

# **Step 8: Show Predictions**
st.subheader("📝 Predicted Prices vs Actual Prices")
results = pd.DataFrame({"Actual Price": y_test.values, "Predicted Price": y_pred})
st.write(results.head())

# **Step 9: User Input for Future Prediction**
st.subheader("📝 Predict Future Prices")
year = st.number_input("Enter Year", min_value=2024, max_value=2030, value=2024)
month = st.number_input("Enter Month", min_value=1, max_value=12, value=1)
day = st.number_input("Enter Day", min_value=1, max_value=31, value=1)
min_price = st.number_input("Enter Min Price", min_value=500, max_value=5000, value=1000)
max_price = st.number_input("Enter Max Price", min_value=500, max_value=5000, value=2000)

# **Step 10: Predict Future Price**
input_data = scaler.transform([[year, month, day, min_price, max_price]])
predicted_price = model.predict(input_data)

st.subheader("📌 Predicted Modal Price")
st.write(f"💰 **Predicted Price: ₹{predicted_price[0]:.2f} per quintal**")

# **Done! 🚀**
