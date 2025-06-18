import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# Load and preprocess dataset for training (ideally done once, then saved)
df = pd.read_csv("personality_dataset.csv")

# Encode categorical features
label_enc = LabelEncoder()
df['Stage_fear'] = label_enc.fit_transform(df['Stage_fear'])
df['Drained_after_socializing'] = label_enc.fit_transform(df['Drained_after_socializing'])
df['Personality'] = label_enc.fit_transform(df['Personality'])

# Split features and target
X = df.drop('Personality', axis=1)
y = df['Personality']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Streamlit app
st.title("Personality Prediction App")
st.write("Fill in the details to predict if someone is an Extrovert or Introvert")

# Input fields
time_spent_alone = st.slider("Time spent alone (hours)", 0, 24, 2)
stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.slider("Number of social events attended per month", 0, 20, 5)
going_outside = st.slider("How often do you go outside? (days/week)", 0, 7, 4)
drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
friends_circle_size = st.slider("How many close friends do you have?", 0, 50, 10)
post_frequency = st.slider("How often do you post on social media? (posts/week)", 0, 20, 5)

# Encode inputs
stage_fear_encoded = 1 if stage_fear == "Yes" else 0
drained_encoded = 1 if drained_after_socializing == "Yes" else 0

# Create input DataFrame
input_data = pd.DataFrame([{
    'Time_spent_Alone': time_spent_alone,
    'Stage_fear': stage_fear_encoded,
    'Social_event_attendance': social_event_attendance,
    'Going_outside': going_outside,
    'Drained_after_socializing': drained_encoded,
    'Friends_circle_size': friends_circle_size,
    'Post_frequency': post_frequency
}])

# Predict button
if st.button("Predict Personality"):
    prediction = model.predict(input_data)[0]
    personality = "Introvert" if prediction == 1 else "Extrovert"
    st.success(f"Predicted Personality: {personality}")
