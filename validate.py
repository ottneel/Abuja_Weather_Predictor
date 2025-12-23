import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels
import pmdarima as pm
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

# CONFIGURATION
EXPERIMENT_NAME = "Abuja_Temp_Validation"
MODEL_NAME = "AbujaTemps"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)

def get_clean_data():
    """Fetches data from DB and resamples to Daily Mean."""
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    
    query = "SELECT timestamp, temperature FROM sensor_data WHERE temperature IS NOT NULL ORDER BY timestamp ASC"
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df['temperature'].resample('D').mean().interpolate(method='time')

def evaluate_params(train_data, test_data, order, seasonal_order, name):
    """Trains on 'Past', Validates on 'Future' (Test Set). Returns MAE."""
    print(f"Testing {name}: {order} x {seasonal_order}")
    
    # 1. Fit on Train Data
    model = SARIMAX(
        train_data, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # 2. Walk-Forward Validation on Test Data
    predictions = []
    history = model_fit
    
    for t in range(len(test_data)):
        # Predict one step ahead
        yhat = history.forecast(steps=1).iloc[0]
        predictions.append(yhat)
        # Update history with real observation (no full refit)
        history = history.append([test_data.iloc[t]], refit=False)
        
    mae = mean_absolute_error(test_data, predictions)
    print(f"Result: MAE = {mae:.4f}")
    return mae, predictions

def evaluate():
    data = get_clean_data()
    if len(data) < 50: 
        print("Not enough data to run validation.")
        return

    # Split: Train on first 80%, Test on last 20%
    split = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:split], data.iloc[split:]
    
    with mlflow.start_run(run_name="Current_vs_New"):
        
        # 1. Current Production Model
        current_mae = float('inf')
        current_order = None
        current_seasonal = None
        
        try:
            print("Fetching current Production model...")
            # Load specifically the model tagged 'Production' in Registry
            prod_model = mlflow.statsmodels.load_model(f"models:/{MODEL_NAME}/Production")
            current_order = prod_model.model.order
            current_seasonal = prod_model.model.seasonal_order
            
            current_mae, current_preds = evaluate_params(train_data, test_data, current_order, current_seasonal, "Current")
        except Exception:
            print("No Production model found. Current is disqualified.")

        # 2. New Auto-ARIMA Search
        print("Running Auto-ARIMA to find better Parameters...")
        auto = pm.auto_arima(
            train_data, start_p=0, start_q=0, max_p=3, max_q=3, m=7,
            seasonal=True, stepwise=True, suppress_warnings=True, error_action='ignore'
        )
        new_mae, new_preds = evaluate_params(train_data, test_data, auto.order, auto.seasonal_order, "New")

        # THE DECISION
        # New Parameters must improve by at least 0.05 MAE to replace Current
        improvement_threshold = 0.05
        
        if current_mae == float('inf'):
            winner = "New (Default)"
            w_order, w_seasonal, w_mae, w_preds = auto.order, auto.seasonal_order, new_mae, new_preds
        elif new_mae < (current_mae - improvement_threshold):
            winner = "New (New Params)"
            w_order, w_seasonal, w_mae, w_preds = auto.order, auto.seasonal_order, new_mae, new_preds
        else:
            winner = "Current (Retain Old)"
            w_order, w_seasonal, w_mae, w_preds = current_order, current_seasonal, current_mae, current_preds

        print(f"\n WINNER: {winner} (MAE: {w_mae:.4f})")

        # 4. LOG RESULTS FOR TRAINING
        mlflow.log_param("winner", winner)
        # CRITICAL: Log these as strings so Training Script can read them
        mlflow.log_param("best_order", str(w_order))
        mlflow.log_param("best_seasonal_order", str(w_seasonal))
        mlflow.log_metric("val_mae", w_mae)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(test_data.index, test_data, label='Actual')
        plt.plot(test_data.index, w_preds, color='red', linestyle='--', label=f'Forecast ({winner})')
        plt.title(f'Validation Winner: {winner}')
        plt.legend()
        plt.savefig("winner.png")
        mlflow.log_artifact("winner.png")
        if os.path.exists("winner.png"): os.remove("winner.png")

if __name__ == "__main__":
    evaluate()