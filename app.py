from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)

# ------------------------------------------
# LOAD DATASET (optional)
# ------------------------------------------
DATA_PATH = "energy_data.csv"
try:
    data = pd.read_csv(DATA_PATH)
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.sort_values("datetime")
except Exception as e:
    print("Could not load CSV:", e)

# ------------------------------------------
# LOAD TRAINED CNN-LSTM MODEL
# ------------------------------------------
# IMPORTANT:
# This model must be trained with 6 input features:
# ['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']
# and target = 'Power demand'
with open("my_cnn_lstm_model.pkl", "rb") as f:
    model = pickle.load(f)


def make_input_array(temp, dew_point, humidity,
                     wind_dir, wind_speed, pressure):
    """
    Build input of shape (1, 30, 6) for the CNN-LSTM model.

    Feature order MUST match training:
    [temp, dwpt, rhum, wdir, wspd, pres]
    """
    base_features = np.array(
        [
            temp,           # temp
            dew_point,      # dwpt
            humidity,       # rhum
            wind_dir,       # wdir
            wind_speed,     # wspd
            pressure,       # pres
        ],
        dtype=np.float32,
    )

    # Repeat same feature vector for 30 time steps
    sequence_30x6 = np.tile(base_features, (30, 1))  # (30, 6)
    input_data = sequence_30x6.reshape(1, 30, 6)     # (1, 30, 6)
    return input_data


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    # default values for form
    form_defaults = {
        "temperature": 25.0,
        "dew_point": 20.0,
        "humidity": 50.0,
        "wind_direction": 180.0,
        "wind_speed": 10.0,
        "pressure": 1013.0,
        "forecast_date": datetime.today().date().isoformat(),
    }

    if request.method == "POST":
        try:
            temperature = float(request.form.get("temperature", 25.0))
            dew_point = float(request.form.get("dew_point", 20.0))
            humidity = float(request.form.get("humidity", 50.0))
            wind_direction = float(request.form.get("wind_direction", 180.0))
            wind_speed = float(request.form.get("wind_speed", 10.0))
            pressure = float(request.form.get("pressure", 1013.0))
            forecast_date = request.form.get(
                "forecast_date",
                datetime.today().date().isoformat()
            )

            # keep values to re-fill the form
            form_defaults.update(
                {
                    "temperature": temperature,
                    "dew_point": dew_point,
                    "humidity": humidity,
                    "wind_direction": wind_direction,
                    "wind_speed": wind_speed,
                    "pressure": pressure,
                    "forecast_date": forecast_date,
                }
            )

            # Build model input (NO power_demand passed in)
            input_data = make_input_array(
                temperature,
                dew_point,
                humidity,
                wind_direction,
                wind_speed,
                pressure,
            )

            # Predict power demand
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
    JSON body should be:

    {
      "temperature": 25,
      "dew_point": 20,
      "humidity": 50,
      "wind_direction": 180,
      "wind_speed": 10,
      "pressure": 1013
    }
    """
    try:
        data_json = request.get_json()

        temperature = float(data_json["temperature"])
        dew_point = float(data_json["dew_point"])
        humidity = float(data_json["humidity"])
        wind_direction = float(data_json["wind_direction"])
        wind_speed = float(data_json["wind_speed"])
        pressure = float(data_json["pressure"])

        input_data = make_input_array(
            temperature,
            dew_point,
            humidity,
            wind_direction,
            wind_speed,
            pressure,
        )

        pred = model.predict(input_data)
        prediction = float(np.ravel(pred)[0])

        return jsonify({"power_demand": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
