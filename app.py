from flask import Flask, request, jsonify
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import date
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Global variables for storing datasets, models, scaler, and encoder
datasets = []
current_date = date.today()
model_filename = "easy-rento-trained_model.joblib"
scaler = StandardScaler()
encoder = OneHotEncoder(drop='first', sparse=False)
gpr_model = None

# Function to get or load the Gaussian Process Regression model
def get_gpr_model():
    global gpr_model
    if gpr_model is None:
        if os.path.exists(model_filename):
            print(f'Loading pre-trained model : {model_filename}')
            gpr_model = joblib.load(model_filename)
        else:
            raise ValueError('Model not trained yet')
    return gpr_model

# Function to preprocess input features (fit_transform)
def preprocess_input_features(X):
    global scaler, encoder
    if scaler is None or encoder is None:
        raise ValueError('Scaler or encoder not initialized')

    # Handle unknown categories during transform
    try:
        X_encoded = encoder.transform(X[['location', 'electricity']])
    except ValueError as e:
        # Handle the error, e.g., log it, raise a more specific exception, or handle it in another way
        raise ValueError(f'Error during transformation: {str(e)}')

    X_scaled = scaler.transform(X[['total_rooms', 'total_bedrooms', 'hotwater', 'terrace', 'markets']])
    X_encoded = encoder.transform(X[['location', 'electricity']])
    return np.concatenate((X_scaled, X_encoded), axis=1)

@app.route('/train_model', methods=['POST'])
def train_model():
    global gpr_model, scaler, encoder
    try:
        # Get input features and target variable from the request
        data = request.json['data']

        # Create a DataFrame from the received data
        df = pd.DataFrame(data)

        # Extract features and target variable
        X = df.drop(columns=['rental_price'])
        y = df['rental_price']

        # Preprocess input features (fit_transform)
        scaler.fit(X[['total_rooms', 'total_bedrooms', 'hotwater', 'terrace', 'markets']])
        encoder.fit(X[['location', 'electricity']])

        X_processed = preprocess_input_features(X)

        # Train the Gaussian Process Regression model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gpr_model.fit(X_processed, y)

        # Save the trained model using joblib
        joblib.dump(gpr_model, model_filename)

        # Save the dataset for future reference
        datasets.append(df)

        return jsonify({'message': 'Model trained successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_price', methods=['POST'])
def predict_price():
    global gpr_model, scaler, encoder
    try:
        # Get input features from the request
        input_features = request.json['features']

        # Get or load the Gaussian Process Regression model
        gpr_model = get_gpr_model()

        # Preprocess input features (transform only)
        input_features_df = pd.DataFrame([input_features])
        input_features_processed = preprocess_input_features(input_features_df)

        # Make a prediction using the loaded model
        predicted_price = gpr_model.predict(input_features_processed)[0]

        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to preprocess input features (fit_transform)
def preprocess_input_features(X):
    global scaler, encoder
    if scaler is None or encoder is None:
        raise ValueError('Scaler or encoder not initialized')
    X_scaled = scaler.transform(X[['total_rooms', 'total_bedrooms', 'hotwater', 'terrace', 'markets']])
    X_encoded = encoder.transform(X[['location', 'electricity']])
    return np.concatenate((X_scaled, X_encoded), axis=1)

# Load data from CSV file
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f'Error loading data from CSV file: {str(e)}')

# Function to train the model
def train_model(data):
    global gpr_model, scaler, encoder
    try:
        # Extract features and target variable
        X = data.drop(columns=['rental_price'])
        y = data['rental_price']

        # Preprocess input features (fit_transform)
        scaler.fit(X[['total_rooms', 'total_bedrooms', 'hotwater', 'terrace', 'markets']])
        encoder.fit(X[['location', 'electricity']])

        X_processed = preprocess_input_features(X)

        # Train the Gaussian Process Regression model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gpr_model.fit(X_processed, y)

        # Save the trained model using joblib
        joblib.dump(gpr_model, model_filename)

        # Save the dataset for future reference
        datasets.append(data)

        return {'message': 'Model trained successfully'}
    except Exception as e:
        raise ValueError(f'Error training model: {str(e)}')

@app.route('/train_model_from_file', methods=['POST'])
def train_model_endpoint():
    try:
        # Get the file path from the request
        file_path = request.json.get('file_path')

        # Load data from the CSV file
        data = load_data(file_path)

        # Train the model
        response = train_model(data)

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

