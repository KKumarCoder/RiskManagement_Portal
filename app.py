import pandas as pd
import numpy as np
from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
file_path = "data/Data.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Encode target variable
df["Predicted Result"] = df["Predicted Result"].map({"Low Risk": 0, "High Risk": 1})

# Selecting features (excluding User ID and target variable)
X = df.drop(columns=["User ID", "Predicted Result"])
y = df["Predicted Result"]

# Train Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get user input from the form
            input_data = [float(request.form[col]) for col in X.columns]
            
            # Convert to DataFrame
            user_df = pd.DataFrame([input_data], columns=X.columns)
            
            # Predict risk
            prediction_value = model.predict(user_df)[0]
            predicted_class = 1 if prediction_value >= 0.5 else 0
            prediction = "High Risk" if predicted_class == 1 else "Low Risk"
        
        except ValueError:
            prediction = "Invalid input. Please enter valid numbers."
    
    return render_template("index.html", prediction=prediction, columns=X.columns)

if __name__ == "__main__":
    app.run(debug=True)
