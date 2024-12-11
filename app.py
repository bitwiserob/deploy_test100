from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model and scaler
def load_model_and_scaler():
    model = tf.keras.models.load_model('housing_lstm_model.keras')
    scaler_params = np.load('scaler_params.npy', allow_pickle=True).item()

    scaler = MinMaxScaler()
    scaler.scale_ = scaler_params['scale_']
    scaler.min_ = scaler_params['min_']
    scaler.data_min_ = scaler_params['data_min_']
    scaler.data_max_ = scaler_params['data_max_']

    return model, scaler

# Load model and scaler globally
model, scaler = load_model_and_scaler()

# Predict price
def predict_price(prices):
    # Convert input to a NumPy array
    prices = np.array(prices)

    # Scale input
    scaled_input = scaler.transform(prices.reshape(-1, 1))
    scaled_input = scaled_input.reshape(1, 10, 1)

    # Make prediction
    prediction = model.predict(scaled_input, verbose=0)

    # Inverse transform prediction
    final_prediction = scaler.inverse_transform(prediction)[0][0]

    return f"${final_prediction:,.2f}"

# Flask endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON payload
        data = request.json
        prices = data.get("prices")

        # Validate the input
        if not prices or len(prices) != 10:
            return jsonify({"error": "Please provide exactly 10 house prices"}), 400

        # Ensure prices are numbers
        prices = [float(price) for price in prices]

        # Make prediction
        predicted_price = predict_price(prices)
        return jsonify({"predicted_price": predicted_price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
