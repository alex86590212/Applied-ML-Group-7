import streamlit as st
import requests

API_URL = "http://localhost:8000/predict-audio"

st.title("Vehicle Sound Classifier")

LABEL_MAP = {
    0: "Airplane",
    1: "Bics",
    2: "Cars",
    3: "Helicopter",
    4: "Motocycles",
    5: "Train",
    6: "Truck",
    7: "Bus"
}

st.markdown("### Your sound can be classified in one of the following classes:")
st.markdown(", ".join(f"*{label}*" for label in LABEL_MAP.values()))


model_type = st.selectbox("Choose a model", ["RNN", "CNN", "Combined"])

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file and model_type:
    if st.button("Predict"):
        with st.spinner("Sending to server and predicting..."):
            files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
            data = {"model_type": model_type}
            try:
                response = requests.post(API_URL, files=files, data=data)
                response.raise_for_status()
                result = response.json()
                st.success(f"Our {result['model']} arhitecture classified your audio file into the *{result['prediction_label']}* category ")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")