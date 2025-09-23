from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("transferiq_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    performance = float(request.form["performance"])
    sentiment = float(request.form["sentiment"])
    injury = int(request.form["injury"])
    contract = int(request.form["contract"])

    features = [[performance, sentiment, injury, contract]]
    prediction = model.predict(features)[0]

    return render_template("index.html", prediction_text=f"Predicted Transfer Value: €{prediction:.2f}M")

if __name__ == "__main__":
    app.run(debug=True)
