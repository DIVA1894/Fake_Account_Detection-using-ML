import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and features
model = joblib.load("fake_account_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load the dataset for visualizations (assuming it's a CSV file)
# Replace the path with the actual path to your dataset
df = pd.read_csv('D:/ML_PROJECT/instagram-fake-detector/final-v1.csv')

# Pie and Bar Chart for Fake vs. Real Accounts
fake_real_count = df['is_fake'].value_counts()
fake_real_count.index = ['Real', 'Fake']

# Generate Pie Chart
st.subheader("Distribution of Fake and Real Accounts")
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(fake_real_count, labels=fake_real_count.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'], startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
st.pyplot(fig)

# Generate Bar Chart
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=fake_real_count.index, y=fake_real_count.values, palette='Set2', ax=ax)
ax.set_title("Fake vs Real Accounts (Bar Graph)")
ax.set_xlabel("Account Type")
ax.set_ylabel("Count")
st.pyplot(fig)

# Model Prediction
st.title("Instagram Fake Account Detector")

input_data = []

st.subheader("Enter Account Details:")

for feature in feature_names:
    value = st.number_input(f"{feature}", step=1.0)
    input_data.append(value)

if st.button("Predict"):
    data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(data)[0]
    
    if prediction == 1:
        st.error("⚠️ This account is predicted to be **FAKE**.")
    else:
        st.success("✅ This account is predicted to be **REAL**.")
