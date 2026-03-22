from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import json
import numpy as np

# Load model and features relative to this file, not the current shell directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'linear_mmm_model.pkl'
FEATURES_PATH = BASE_DIR / 'mmm_model_features.json'

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    feature_names = json.load(f)

app = Flask(__name__)


def build_feature_vector(row):
    return [float(row.get(feat, 0)) for feat in feature_names]

@app.route('/')
def home():
    return "MMM Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if isinstance(data, dict):
        records = [data]
        single_record = True
    elif isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        records = data
        single_record = False
    else:
        return jsonify({
            'error': 'Request body must be a JSON object or a non-empty JSON array of objects.'
        }), 400

    try:
        matrix = np.array([build_feature_vector(row) for row in records], dtype=float)
    except (TypeError, ValueError) as exc:
        return jsonify({
            'error': 'All feature values must be numeric.',
            'details': str(exc)
        }), 400

    predictions = [float(value) for value in model.predict(matrix)]

    if single_record:
        return jsonify({'predicted_revenue': predictions[0]})

    return jsonify({
        'predictions': [
            {'predicted_revenue': prediction}
            for prediction in predictions
        ]
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, use_reloader=False)
