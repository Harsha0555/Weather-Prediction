import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

df = pd.read_csv('data2.csv')

features = ['Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
target = "Weather"

x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

st.title("Weather Prediction")

st.sidebar.header("Dataset Information")
st.sidebar.info("This application uses a K-Nearest Neighbors classifier to predict the Weather")

st.subheader("Dataset")
st.write(df)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)*100
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

st.sidebar.header("User Input")
temp_c = st.sidebar.slider("Temperature (Celsius)", min_value=-10, max_value=40, value=20)
rel_hum = st.sidebar.slider("Relative Humidity (%)", min_value=20, max_value=100, value=50)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", min_value=0, max_value=80, value=30)
visibility = st.sidebar.slider("Visibility (km)", min_value=0, max_value=50, value=20)
press_kpa = st.sidebar.slider("Pressure (kPa)", min_value=90, max_value=105, value=100)

user_input = [[temp_c, rel_hum, wind_speed, visibility, press_kpa]]
prediction = model.predict(user_input)

st.subheader("Prediction")
st.write(f"The predicted cluster for the given input is: {prediction[0]}")


# streamlit run frun2.py
