# 🌦️ Rainfall Prediction using Machine Learning

A simple yet powerful Flask-based web application that predicts whether it will rain tomorrow based on input weather conditions. This project leverages machine learning algorithms and provides a user-friendly interface for rainfall prediction.

---

## 📌 Features

- Uses real-world historical weather data
- Implements multiple ML algorithms like Random Forest, Decision Tree, etc.
- Web interface built with Flask
- Displays output as "Sunny Day" or "Rainy Day"
- Suitable for farmers, planners, or educational purposes

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS (custom style)
- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn
- **Model Used**: RandomForestClassifier (best accuracy)

---

## 🧠 Machine Learning Workflow

- Dataset: `weatherAUS.csv` (cleaned and preprocessed)
- Features: Temperature, Humidity, Pressure, Wind, RainToday
- Target: RainTomorrow (Yes/No)
- Best accuracy achieved: **86%** using Random Forest
- Model saved as `rain_model2.sav`

---

## 🔍 Screenshots

### 🔹 Homepage
![Homepage](screenshots/homepage.png)

### 🔹 Input Form
![Input](screenshots/input_form.png)

### 🔹 Rainy Prediction
![Rainy](screenshots/result_rainy.png)

### 🔹 Sunny Prediction
![Sunny](screenshots/result_sunny.png)

---

## 🚀 How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/RainfallPredictionML.git
   cd RainfallPredictionML
