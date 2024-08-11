import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load model and scaler
model = pickle.load(open(r'models\modelPrediction.pkl', 'rb'))
standard_scaler = pickle.load(open(r'models\standardScalar.pkl', 'rb'))

fuel_type_mapping = {
    0: "Diesel",
    1: "LPG",
    2: "Petrol"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            name_encoded = int(request.form.get('name_encoded'))
            year = int(request.form.get('year'))
            Price = float(request.form.get('Price'))
            kms_driven = float(request.form.get('kms_driven'))

            # Validate inputs
            if any(v is None for v in [name_encoded, year, Price, kms_driven]):
                return render_template('home.html', result="Invalid input!")

            # Scale the new data
            new_data_scaled = standard_scaler.transform([[name_encoded, year, Price, kms_driven]])

            # Predict the fuel type
            result = model.predict(new_data_scaled)
            predicted_fuel_type = fuel_type_mapping.get(result[0], "Unknown")

            output_label = f"Fuel Type is: {predicted_fuel_type}"
        except Exception as e:
            output_label = f"Error: {str(e)}"
        
        return render_template('home.html', result=output_label)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
