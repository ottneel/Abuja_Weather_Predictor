import os
import pandas as pd
import mlflow
import mlflow.statsmodels
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore') # Silence innocent warnings
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

        # Hyperparameters (ARIMA p,d,q)
        # p=1 (Autoregression), d=1 (Trend), q=1 (Moving Average)
        order = (1, 1, 1)
        print(f"Training ARIMA{order}...")
        
        mlflow.log_param("order", order)
        
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        
        # Log Performance
        aic = model_fit.aic
        mlflow.log_metric("aic", aic)
        print(f"Training Complete. AIC Score: {aic}")
        
        # Save Model
        model_info = mlflow.statsmodels.log_model(
            model_fit, 
            artifact_path="model", 
            registered_model_name="AbujaTemps"
        )
        
        print(f"Model saved to MLflow as version '{model_info.registered_model_version}'")

if __name__ == "__main__":
    train()