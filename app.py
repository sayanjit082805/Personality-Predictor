import streamlit as st
import pandas as pd
from keras.models import load_model
import joblib
import numpy as np

model = load_model("personality_model.keras")
model_columns = joblib.load("model_columns.pkl")
class_labels = joblib.load("class_labels.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Personality Predictor",
    page_icon=":robot:",
    initial_sidebar_state="expanded",
)

st.title("Personality Predictor")

st.subheader(
    "Predict the type of personality based on behavioral data.",
    divider="gray",
)

st.write(
    """
   A deep learning model to classify extrovert vs introvert personality types using behavioral data.      
"""
)

st.sidebar.header("Input Parameters")


def input_features():
    time = st.sidebar.slider(
        "Time Spent Alone", min_value=0.0, max_value=20.0, value=5.0, step=0.1
    )
    fear = st.sidebar.selectbox(
        "Stage Fear", options=["Yes", "No"], index=0
    )
    social = st.sidebar.slider(
        "Social Events", min_value=0.0, max_value=15.0, value=5.0, step=0.1
    )
    outside = st.sidebar.slider(
        "Time Spent Outside", min_value=0.0, max_value=10.0, value=2.0, step=1.0
    )
    friends = st.sidebar.slider(
        "Friends Circle Size", min_value=0.0, max_value=20.0, value=5.0, step=1.0
    )
    post = float(
        st.sidebar.slider("Social Media Post Frequency", min_value=0.0, max_value=15.0, value=4.0, step=1.0)
    )
   
   # features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

    features = pd.DataFrame(
        [
            {
                "Time_spent_Alone": time,
                "Stage_fear": fear,
                "Social_event_attendance": social,
                "Going_outside": outside,
                "Friends_circle_size": friends,
                "Post_frequency": post,
            }
        ]
    )

    features_encoded = pd.get_dummies(features)
    features_encoded = features_encoded.reindex(columns=model_columns, fill_value=0)

    features_scaled = scaler.transform(features_encoded)

    return features_scaled


input = input_features()


if st.button("Predict"):
    prediction = model.predict(input)
    prediction_class = np.argmax(prediction, axis=1)[0]
    st.success(
        f"Predicted personality: **{class_labels[prediction_class]}**",
    )
