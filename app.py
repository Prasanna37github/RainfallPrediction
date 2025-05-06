from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("rain_model2.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect input values from form
            data = [
                float(request.form['mintemp']),
                float(request.form['maxtemp']),
                float(request.form['rainfall']),
                float(request.form['windgustspeed']),
                float(request.form['windspeed9am']),
                float(request.form['windspeed3pm']),
                float(request.form['humidity9am']),
                float(request.form['humidity3pm']),
                float(request.form['pressure9am']),
                float(request.form['pressure3pm']),
                float(request.form['temp9am']),
                float(request.form['temp3pm'])
            ]

            # Reshape and scale input
            scaled_input = scaler.transform([data])
            
            # Predict using trained model
            prediction = model.predict(scaled_input)[0]

            # Route based on prediction
            if prediction == 1:
                return render_template('rainy.html')
            else:
                return render_template('sunny.html')

        except Exception as e:
            return f"Error: {e}"

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
