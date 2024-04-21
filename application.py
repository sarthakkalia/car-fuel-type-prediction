import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

model=pickle.load(open('models\modelPrediction.pkl','rb'))
standard_scaler=pickle.load(open('models\standardScalar.pkl','rb'))

fuel_type_mapping = {
    0: "Diesel",
    1: "LPG",
    2: "Petrol"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        name_encoded=int(request.form.get('name_encoded'))
        company_encoded = int(request.form.get('company_encoded'))
        year = int(request.form.get('year'))
        Price = float(request.form.get('Price'))
        kms_driven = float(request.form.get('kms_driven'))

        new_data_scaled=standard_scaler.transform([[name_encoded,company_encoded,year,Price,kms_driven]])
        result=model.predict(new_data_scaled)

        predicted_fuel_type = fuel_type_mapping[result[0]]

        output_label = "Fuel Type is: " + predicted_fuel_type

        return render_template('home.html', result=output_label)
    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")