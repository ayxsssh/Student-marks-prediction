from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

# Import all models and metrics just as the user did
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)
CORS(app)

# Global dictionaries to store models and metrics
models = {}
metrics = {}
# For models that require preprocessing specifically
preprocessors = {}

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mse, rmse, mae

def initialize_models():
    # Load dataset
    data = pd.read_csv('student_exam_scores.csv')

    # Base Input (X) and Output (y)
    X = data[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']]
    y = data['exam_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Linear Regression...")
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    models['linear'] = model_lr
    y_test_pred = model_lr.predict(X_test)
    r2, mse, rmse, mae = evaluate(y_test, y_test_pred)
    metrics['linear'] = {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}

    print("Training Polynomial Regression...")
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train_poly)
    models['polynomial'] = model_poly
    preprocessors['polynomial'] = poly
    y_test_pred_poly = model_poly.predict(X_test_poly)
    r2, mse, rmse, mae = evaluate(y_test_poly, y_test_pred_poly)
    metrics['polynomial'] = {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}

    print("Training KNN...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train_scaled)
    models['knn'] = knn
    preprocessors['knn'] = scaler
    y_test_pred_knn = knn.predict(X_test_scaled)
    r2, mse, rmse, mae = evaluate(y_test_scaled, y_test_pred_knn)
    metrics['knn'] = {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}

    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42) 
    # Skipping grid search to make startup faster, using best expected params
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    y_test_pred_rf = rf.predict(X_test)
    r2, mse, rmse, mae = evaluate(y_test, y_test_pred_rf)
    metrics['random_forest'] = {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}

    print("Training Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=4, random_state=42)
    dt.fit(X_train, y_train)
    models['decision_tree'] = dt
    y_test_pred_dt = dt.predict(X_test)
    r2, mse, rmse, mae = evaluate(y_test, y_test_pred_dt)
    metrics['decision_tree'] = {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}

    # print("Training XGBoost...")
    # xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
    # xgb.fit(X_train, y_train)
    # models['xgboost'] = xgb
    # y_test_pred_xgb = xgb.predict(X_test)
    # r2, mse, rmse, mae = evaluate(y_test, y_test_pred_xgb)
    # metrics['xgboost'] = {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}

    print("All models trained successfully!")

# Train models on import
initialize_models()

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        "models": list(models.keys()),
        "metrics": metrics
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_name = data.get('model', 'linear')
        features = data.get('features') # [hours_studied, sleep_hours, attendance_percent, previous_scores]
        
        if not features or len(features) != 4:
            return jsonify({"error": "Please provide exactly 4 features."}), 400
        if model_name not in models:
            return jsonify({"error": "Invalid model name."}), 400
        
        input_data = pd.DataFrame([features], columns=['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores'])
        
        # Apply model specific preprocessing
        if model_name == 'polynomial':
            input_data = preprocessors['polynomial'].transform(input_data)
        elif model_name == 'knn':
            input_data = preprocessors['knn'].transform(input_data)
            
        prediction = models[model_name].predict(input_data)[0]
        
        # Cap logic just in case model overshoots
        prediction = max(0, min(100, float(prediction)))
        
        return jsonify({
            "model": model_name,
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
