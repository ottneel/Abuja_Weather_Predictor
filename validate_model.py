import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore') # Silence innocent warnings
load_dotenv()

# Setup MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
mlflow.set_experiment("Abuja_Temp_Validation")

def get_clean_data():
    """
    Connects to your DB and pulls the data for validation.
    """
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    
    print("Fetching data for validation...")
    query = """
        SELECT timestamp, temperature 
        FROM sensor_data 
        WHERE temperature IS NOT NULL
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, engine)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Resample to Daily Mean to match training logic
    # This ensures consistency between validation and production
    df = df['temperature'].resample('D').mean().interpolate(method='time')
    
    return df

def run_walk_forward_validation():
    data = get_clean_data()
    
    if len(data) < 50:
        print("CRITICAL ERROR: Not enough data to validate. Need at least 50 days.")
        return

    # 1. Define Split (Train on first 80%, Validate on last 20%)
    # This simulates "The Past" (Train) vs "The Future" (Test)
    split_point = int(len(data) * 0.8)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    print(f"Training on {len(train_data)} days. Validating on {len(test_data)} days.")

    with mlflow.start_run(run_name="Walk_Forward_Validation"):
        
        # 2. MODEL SELECTION LOGIC
        # We need to decide: Do we test an existing model, or find a new one?
        
        print("Attempting to load latest Production model...")
        model_fit = None
        
        try:
            # OPTION A: Try to load existing model (Established Project)
            model_uri = "models:/AbujaTemps/latest"
            loaded_model = mlflow.statsmodels.load_model(model_uri)
            
            print(f"Found existing model! Testing parameters: {loaded_model.model.order}")
            
            # We construct a NEW model instance using the OLD parameters (The Recipe)
            # but trained ONLY on the 80% data (The Ingredients)
            model = SARIMAX(
                train_data,
                order=loaded_model.model.order,
                seasonal_order=loaded_model.model.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)

        except Exception as e:
            # OPTION B: COLD START (Fresh Project)
            # If no model exists in MLflow, we must find the best parameters ourselves
            # using only the training data available.
            print(f"No production model found ({e}). Running 'Cold Start' Auto-ARIMA...")
            
            auto_model = pm.auto_arima(
                train_data,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                m=7,              # Weekly seasonality
                seasonal=True,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            print(f"Cold Start Best Params: {auto_model.order} | Seasonal: {auto_model.seasonal_order}")
            
            # Fit the model with these discovered parameters
            model = SARIMAX(
                train_data, 
                order=auto_model.order, 
                seasonal_order=auto_model.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)

        # 3. The Walk-Forward Loop
        # We simulate moving through the test set one day at a time.
        predictions = []
        print("Starting Walk-Forward Loop...")
        
        for t in range(len(test_data)):
            # A. Predict ONE step ahead (Tomorrow)
            yhat = model_fit.forecast(steps=1).iloc[0]
            predictions.append(yhat)
            
            # B. Get the Real Value (Truth)
            obs = test_data.iloc[t]
            
            # C. Update the Model (Walk Forward)
            # refit=False updates the history without re-running the heavy math
            model_fit = model_fit.append([obs], refit=False)

        # 4. Calculate Metrics
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        
        print(f"\n=== Validation Results ===")
        print(f"MAE:  {mae:.4f} °C")
        print(f"RMSE: {rmse:.4f} °C")
        
        # 5. Log to MLflow
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmse", rmse)
        
        # 6. Visualize
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data, label='Actual Temperature')
        plt.plot(test_data.index, predictions, color='red', linestyle='--', label='Walk-Forward Forecast')
        plt.title(f'Walk-Forward Validation (MAE: {mae:.2f}°C)')
        plt.legend()
        plt.grid(True)
        
        # Save plot to MLflow
        plot_path = "validation_plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print("Validation plot saved to MLflow.")
        
        # Clean up local file
        if os.path.exists(plot_path):
            os.remove(plot_path)

if __name__ == "__main__":
    run_walk_forward_validation()