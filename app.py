from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)

# --------------------------------------------------
# LOAD DATASET (optional – only if you want it later)
# --------------------------------------------------
DATA_PATH = "energy_data.csv"
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

# Day encoding map
DAY_MAP = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def make_input_array(
    temperature,
    humidity,
    wind_speed,
    pressure,
    wind_direction,
    dewpoint,
    hour,
    day_of_week,
):
    """
    Build input of shape (1, 30, N) for the CNN-LSTM model.
    Adjust the feature order to match how the model was trained.
    """
    day_encoded = DAY_MAP[day_of_week]

    # Example ordering: [temp, hum, wind_speed, pressure, wind_dir, dewpoint, hour, day, dummy1, dummy2]
    base_features = np.array(
        [
            temperature,
            humidity,
            wind_speed,
            pressure,
            wind_direction,
            dewpoint,
            hour,
            day_encoded,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    sequence_30xN = np.tile(base_features, (30, 1))   # (30, num_features)
    input_data = sequence_30xN.reshape(1, 30, -1)     # (1, 30, num_features)
    return input_data


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    # defaults for form repopulation
    form_defaults = {
        "temperature": 25,
        "humidity": 50,
        "wind_speed": 10,
        "pressure": 1013,         # hPa default
        "wind_direction": 0,      # degrees
        "dewpoint": 15,           # °C
        "hour": 12,
        "day_of_week": "Mon",
        "forecast_date": datetime.today().date().isoformat(),
    }

    if request.method == "POST":
        try:
            temperature = float(request.form.get("temperature", 25))
            humidity = float(request.form.get("humidity", 50))
            wind_speed = float(request.form.get("wind_speed", 10))
            pressure = float(request.form.get("pressure", 1013))
            wind_direction = float(request.form.get("wind_direction", 0))
            dewpoint = float(request.form.get("dewpoint", 15))
            hour = int(request.form.get("hour", 12))
            day_of_week = request.form.get("day_of_week", "Mon")
            forecast_date = request.form.get("forecast_date")

            # keep values in form if there is an error
            form_defaults.update(
                {
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "pressure": pressure,
                    "wind_direction": wind_direction,
                    "dewpoint": dewpoint,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "forecast_date": forecast_date,
                }
            )

            # Build model input
            input_data = make_input_array(
                temperature,
                humidity,
                wind_speed,
                pressure,
                wind_direction,
                dewpoint,
                hour,
                day_of_week,
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
        active_tab="predict",
    )


@app.route("/guidance")
def guidance():
    return render_template("guidance.html", active_tab="guidance")


@app.route("/analysis")
def analysis():
    return "<h1 style='text-align:center;margin-top:40px;'>Analysis Page Coming Soon</h1>"


@app.route("/awareness")
def awareness():
    return "<h1 style='text-align:center;margin-top:40px;'>Energy Awareness Page Coming Soon</h1>"


# ---------- JSON API endpoint ----------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expected JSON body:
    {
      "temperature": 25,
      "humidity": 50,
      "wind_speed": 10,
      "pressure": 1013,
      "wind_direction": 180,
      "dewpoint": 15,
      "hour": 12,
      "day_of_week": "Mon"
    }
    """
    try:
        data_json = request.get_json()
        temperature = float(data_json["temperature"])
        humidity = float(data_json["humidity"])
        wind_speed = float(data_json["wind_speed"])
        pressure = float(data_json["pressure"])
        wind_direction = float(data_json["wind_direction"])
        dewpoint = float(data_json["dewpoint"])
        hour = int(data_json["hour"])
        day_of_week = data_json["day_of_week"]

        input_data = make_input_array(
            temperature,
            humidity,
            wind_speed,
            pressure,
            wind_direction,
            dewpoint,
            hour,
            day_of_week,
        )

        pred = model.predict(input_data)
        prediction = float(np.ravel(pred)[0])

        return jsonify({"prediction_kwh": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
