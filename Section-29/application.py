from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predictdata", methods=["POST"])
def predict_data():

    # Fetch data from form
    Temperature = float(request.form.get("Temperature"))
    RH = float(request.form.get("RH"))
    WS = float(request.form.get("WS"))
    Rain = float(request.form.get("Rain"))
    FFMC = float(request.form.get("FFMC"))
    DMC = float(request.form.get("DMC"))
    ISI = float(request.form.get("ISI"))
    Classes = float(request.form.get("Classes"))
    Region = float(request.form.get("Region"))

    # Scale data
    new_data_scaled = standard_scaler.transform(
        [[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]]
    )

    # Prediction
    result = ridge_model.predict(new_data_scaled)

    return render_template("home.html", result=round(result[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
