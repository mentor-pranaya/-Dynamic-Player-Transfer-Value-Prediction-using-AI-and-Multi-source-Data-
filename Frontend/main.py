import sys
print(sys.executable)
import streamlit as st
import requests

st.title("TransferIQ - Player Transfer Value Prediction")

performance = st.text_input("Enter performance stats (comma-separated)")
sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.0)
injury = st.number_input("Injury Count", 0)
contract = st.number_input("Remaining Contract (months)", 0)

if st.button("Predict Value"):
    features = [float(x) for x in performance.split(",")]
    response = requests.post("http://127.0.0.1:8000/predict",
                             json={"performance": features, 
                                   "sentiment": sentiment,
                                   "injury": injury,
                                   "contract": contract})
    st.success(f"Predicted Transfer Value: €{response.json()['predicted_value']:.2f}M")
