import pmdarima as pm
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import pandas as pd
import mlflow
import mlflow.statsmodels
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore') # Silence warnings
load_dotenv()

# Setup MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
mlflow.set_experiment("Abuja_Temp_Forecast")

def get_training_data():
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    
    print("Fetching training data from DB...")
    
    # Grab all data (Historical + New API data)
    query = """
        SELECT timestamp, temperature 
        FROM sensor_data
        WHERE temperature IS NOT NULL 
        ORDER BY timestamp ASC
    """
    
    df = pd.read_sql(query, engine)
    
    # Essential Time-Series Steps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Resample to Daily Mean (Smoothes out multiple readings per day)
    daily_df = df['temperature'].resample('D').mean()
    
    # Fill any tiny gaps that might remain
    daily_df = daily_df.interpolate(method='time')
    
    print(f"Loaded {len(daily_df)} days of training data.")
    return daily_df


def train():
    with mlflow.start_run():
        data = get_training_data()
        
        if data.empty:
            print("CRITICAL ERROR: No data found.")
            return

        print("Searching for optimal ARIMA parameters...")
        
        # 1. AUTO-ARIMA: Find the best parameters automatically
        # m=7 captures weekly patterns.
        auto_model = pm.auto_arima(
            data,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            m=7,              # Monthly/Weekly seasonality check
            seasonal=True,    # Enable Seasonal ARIMA
            d=None,           # Let model figure out differencing
            trace=True,       # Print progress
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True
        )

        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order
        
        print(f"Optimal Parameters Found: Order={best_order}, Seasonal={best_seasonal_order}")
        
        # Log the discovered parameters to MLflow
        mlflow.log_param("order", best_order)
        mlflow.log_param("seasonal_order", best_seasonal_order)
        
        # 2. TRAIN FINAL MODEL (Using Statsmodels for consistency)
        # We retrain using statsmodels to ensure compatibility with your existing inference code
        
        
        model = SARIMAX(
            data, 
            order=best_order, 
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        
        # 3. LOG METRICS
        aic = model_fit.aic
        bic = model_fit.bic
        mlflow.log_metric("aic", aic)
        mlflow.log_metric("bic", bic)
        
        print(f"Training Complete. AIC: {aic:.2f} | BIC: {bic:.2f}")
        
        # 4. SAVE MODEL
        # We log the statsmodels object directly
        model_info = mlflow.statsmodels.log_model(
            model_fit, 
            artifact_path="model", 
            registered_model_name="AbujaTemps"
        )
        
        print(f"Model saved to MLflow as version '{model_info.registered_model_version}'")

if __name__ == "__main__":
    train()