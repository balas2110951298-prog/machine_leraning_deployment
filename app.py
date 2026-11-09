
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Home endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Diabetes Prediction API is running!',
        'prediction_endpoint': '/predict'
    })


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        features = data['features']

        if len(features) != 8:
            return jsonify({
                'error': 'Please provide exactly 8 features'
            }), 400

        features_arr = np.array(features, dtype=float).reshape(1, -1)

        scaled_features = scaler.transform(features_arr)

        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'probability_of_diabetes': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 10000)),
        debug=False
    )
