
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")
columns = joblib.load("house_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        location = request.form["location"].strip()
        area = float(request.form["area"])
        bhk = int(request.form["bhk"])
        bath = int(request.form["bath"])

        if area <= 0 or bhk <= 0 or bath <= 0:
            raise ValueError("Values must be positive")

        
        input_data = pd.DataFrame(
            0, index=[0], columns=columns
        )

        input_data["total_sqft"] = area
        input_data["bhk"] = bhk
        input_data["bath"] = bath

        
        location_column = "location_" + location

        if location_column not in input_data.columns:
            location_column = "location_other"

        if location_column in input_data.columns:
            input_data[location_column] = 1

        
        prediction = model.predict(input_data)[0]

        if prediction <= 0:
            raise ValueError("Invalid prediction")

        return render_template(
            "index.html",
            prediction_text=f"₹ {prediction:.2f} Lakhs"
        )

    except (ValueError, TypeError):
        return render_template(
            "index.html",
            error_text="Please enter valid positive values."
        )

if __name__ == "__main__":
    app.run(debug=True)
