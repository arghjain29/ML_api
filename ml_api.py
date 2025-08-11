from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models at startup
model_a = joblib.load("models/model_a.pkl")
model_b = joblib.load("models/model_b.pkl")
model_c = joblib.load("models/model_c.pkl")

@app.route("/", methods=["GET"])
def health():
    return "ML API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        machine_id = data.get("machineId")

        # Select model and features
        if machine_id == "MachineA":
            features = [data["temperature"], data["vibration"]]
            model = model_a
        elif machine_id == "MachineB":
            features = [data["pressure"], data["rpm"]]
            model = model_b
        elif machine_id == "MachineC":
            features = [data["humidity"], data["temperature"], data["sound_db"]]
            model = model_c
        else:
            return jsonify({"error": "Invalid machineId"}), 400

        # Predict
        X = np.array([features])
        pred = model.predict(X)[0]  # -1 = anomaly, 1 = normal
        result = "anomaly" if pred == -1 else "normal"

        return jsonify({
            "machineId": machine_id,
            "result": result,
            "features": features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
