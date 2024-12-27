from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import category_encoders as ce

app = Flask(__name__)

# Load the trained model
model = joblib.load('xgboost_model.pkl')

# Define a function to preprocess the input data
def preprocess_input(data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame(data, index=[0])
    
    # Encode the 'state' column and other necessary preprocessing steps
    he = ce.HashingEncoder(cols='state')
    input_df = he.fit_transform(input_df)
    
    # Get dummy variables and drop any columns as done during training
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Drop the columns that were dropped during training
    columns_to_drop = ["voice_mail_plan_yes", "total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge"]
    for col in columns_to_drop:
        if col in input_df.columns:
            input_df.drop(columns=[col], inplace=True)
            
    # Ensure input columns match model training columns
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    
    return input_df[model.feature_names_in_]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = {
        "state": request.form['state'],
        "account_length": int(request.form['account_length']),
        "area_code": int(request.form['area_code']),
        "international_plan": request.form['international_plan'],
        "voice_mail_plan": request.form['voice_mail_plan'],
        "number_vmail_messages": int(request.form['number_vmail_messages']),
        "total_day_minutes": float(request.form['total_day_minutes']),
        "total_day_calls": int(request.form['total_day_calls']),
        "total_eve_minutes": float(request.form['total_eve_minutes']),
        "total_eve_calls": int(request.form['total_eve_calls']),
        "total_night_minutes": float(request.form['total_night_minutes']),
        "total_night_calls": int(request.form['total_night_calls']),
        "total_intl_minutes": float(request.form['total_intl_minutes']),
        "total_intl_calls": int(request.form['total_intl_calls']),
        "number_customer_service_calls": int(request.form['number_customer_service_calls']),
    }
    
    # Preprocess the input data
    input_data = preprocess_input(data)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_text = "Khách hàng có khả năng rời bỏ" if prediction == 1 else "Khách hàng không có khả năng rời bỏ"
    
    return render_template('index.html', prediction_text=f'Dự đoán: {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
