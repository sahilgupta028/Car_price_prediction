from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

# Initialize the StandardScaler
scaler = StandardScaler()

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel = 0
    try:
        if request.method == 'POST':
            # Retrieve and process form data
            Year = int(request.form['Year'])
            Present_Price = float(request.form['Present_Price'])
            Kms_Driven = int(request.form['Kms_Driven'])
            Kms_Driven2 = np.log(Kms_Driven)
            Owner = int(request.form['Owner'])
            
            Fuel_Type_Petrol = request.form['Fuel_Type_Petrol']
            if Fuel_Type_Petrol == 'Petrol':
                Fuel_Type_Petrol = 1
                Fuel_Type_Diesel = 0
            else:
                Fuel_Type_Petrol = 0
                Fuel_Type_Diesel = 1
            
            Year = 2020 - Year
            
            Seller_Type_Individual = request.form['Seller_Type_Individual']
            if Seller_Type_Individual == 'Individual':
                Seller_Type_Individual = 1
            else:
                Seller_Type_Individual = 0
            
            Transmission_Manual = request.form['Transmission_Manual']
            if Transmission_Manual == 'Manual':
                Transmission_Manual = 1
            else:
                Transmission_Manual = 0
            
            # Prepare the feature array
            features = np.array([[Present_Price, Kms_Driven2, Owner, Year,
                                  Fuel_Type_Diesel, Fuel_Type_Petrol,
                                  Seller_Type_Individual, Transmission_Manual]])
            
            # Standardize features if required (commented out if not needed)
            # features = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features)
            output = round(prediction[0], 2)
            
            # Render the result
            if output < 0:
                return render_template('index.html', prediction_text="Sorry, you cannot sell this car")
            else:
                return render_template('index.html', prediction_text=f"You can sell the car at ₹{output}")
        else:
            return render_template('index.html')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
