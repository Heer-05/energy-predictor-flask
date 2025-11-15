# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)

# --------------------------------------------------
# LOAD DATASET (optional – only if you want it later)
# --------------------------------------------------
DATA_PATH = "energy_data.csv"  # change if needed
try:
    data = pd.read_csv(DATA_PATH)
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.sort_values("datetime")
except Exception as e:
    print("Could not load CSV:", e)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
with open("my_cnn_lstm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Day encoding map (same as Streamlit)
DAY_MAP = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def make_input_array(temperature, humidity, wind_speed, hour, day_of_week):
    """Build input of shape (1, 30, 7) like in your Streamlit code."""
    day_encoded = DAY_MAP[day_of_week]

    # same as your Streamlit: 5 real + 2 dummy features
    base_features = np.array(
        [temperature, humidity, wind_speed, hour, day_encoded, 0.0, 0.0],
        dtype=np.float32,
    )

    sequence_30x7 = np.tile(base_features, (30, 1))  # (30,7)
    input_data = sequence_30x7.reshape(1, 30, 7)     # (1,30,7)
    return input_data


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    # default values for form repopulation
    form_defaults = {
        "temperature": 25,
        "humidity": 50,
        "wind_speed": 10,
        "hour": 12,
        "day_of_week": "Mon",
        "forecast_date": datetime.today().date().isoformat(),
    }

    if request.method == "POST":
        try:
            temperature = float(request.form.get("temperature", 25))
            humidity = float(request.form.get("humidity", 50))
            wind_speed = float(request.form.get("wind_speed", 10))
            hour = int(request.form.get("hour", 12))
            day_of_week = request.form.get("day_of_week", "Mon")
            forecast_date = request.form.get("forecast_date")

            form_defaults.update(
                {
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "forecast_date": forecast_date,
                }
            )

            # Build model input
            input_data = make_input_array(
                temperature, humidity, wind_speed, hour, day_of_week
            )

            # Predict
            pred = model.predict(input_data)
            prediction = float(np.ravel(pred)[0])

        except Exception as e:
            error = f"Prediction Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form=form_defaults,
    )


# ---------- Optional: JSON API endpoint ----------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Example JSON POST body:
    {
      "temperature": 25,
      "humidity": 50,
      "wind_speed": 10,
      "hour": 12,
      "day_of_week": "Mon"
    }
    """
    try:
        data_json = request.get_json()
        temperature = float(data_json["temperature"])
        humidity = float(data_json["humidity"])
        wind_speed = float(data_json["wind_speed"])
        hour = int(data_json["hour"])
        day_of_week = data_json["day_of_week"]

        input_data = make_input_array(
            temperature, humidity, wind_speed, hour, day_of_week
        )
        pred = model.predict(input_data)
        prediction = float(np.ravel(pred)[0])

        return jsonify({"prediction_kwh": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
