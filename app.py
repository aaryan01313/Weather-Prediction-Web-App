from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

#Load Models and Scaler
print("Loading models and scaler...")
with open('weather_xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('weather_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('weather_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Everything loaded successfully!")

@app.route('/')
def home():
    return "Weather Forecasting API is running smoothly! 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        #array to scale raw inputs (First value is a dummy 0.0 for temperature)
        raw_features_for_scaling = [
            0.0, 
            float(data['wind_kph']), float(data['pressure_mb']), float(data['precip_mm']),
            float(data['humidity']), float(data['visibility_km']), float(data['uv_index']),
            float(data['air_quality_PM2.5']), float(data['air_quality_PM10'])
        ]

        # Columns used during normalization
        cols_to_normalize = ['temperature_celsius', 'wind_kph', 'pressure_mb', 'precip_mm', 'humidity', 'visibility_km', 'uv_index', 'air_quality_PM2.5', 'air_quality_PM10']
        df_to_scale = pd.DataFrame([raw_features_for_scaling], columns=cols_to_normalize)
        
        #scaling
        scaled_values = scaler.transform(df_to_scale)[0]

        #final input for the model 
        model_input = [
            scaled_values[1], scaled_values[2], scaled_values[3], scaled_values[4],
            scaled_values[5], scaled_values[6], scaled_values[7], scaled_values[8],
            int(data['month']), int(data['day']), int(data['hour'])
        ]

        feature_names = ['wind_kph', 'pressure_mb', 'precip_mm', 'humidity', 'visibility_km', 'uv_index', 'air_quality_PM2.5', 'air_quality_PM10', 'month', 'day', 'hour']
        df_final_input = pd.DataFrame([model_input], columns=feature_names)

        # predictions from models
        xgb_pred = xgb_model.predict(df_final_input)[0]
        rf_pred = rf_model.predict(df_final_input)[0]
        scaled_temp_pred = (xgb_pred + rf_pred) / 2

        #Inverse transform to get original temperature
        dummy_for_inverse = np.zeros((1, 9))
        dummy_for_inverse[0, 0] = scaled_temp_pred 
        
        unscaled_result = scaler.inverse_transform(dummy_for_inverse)
        final_temperature = unscaled_result[0, 0]

        return jsonify({
            'prediction': round(float(final_temperature), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)