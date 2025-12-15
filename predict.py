import os
import pandas as pd
import mlflow.statsmodels
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Setup MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))

def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

def generate_forecast():
    model_name = "AbujaTemps"
    print(f"Loading latest model for '{model_name}'...")
    
    try:
        # Load the latest Production or Staging model (or just latest version)
        # Note: If you haven't transitioned a model to 'Production' in UI, use 'latest'
        model_uri = f"models:/{model_name}/latest"
        loaded_model = mlflow.statsmodels.load_model(model_uri)
        
        # Forecast 7 days into the future
        steps = 7
        forecast = loaded_model.forecast(steps=steps)
        
        # Create dates for the forecast
        today = datetime.now().date()
        forecast_dates = [today + timedelta(days=i) for i in range(1, steps + 1)]
        
        # Prepare DataFrame for DB
        forecast_df = pd.DataFrame({
            'forecast_date': forecast_dates,
            'predicted_temperature': forecast.values,
            'model_version': 'latest', # Ideally you'd grab the actual version number
            'created_at': datetime.now()
        })
        
        print("Predictions generated:")
        print(forecast_df[['forecast_date', 'predicted_temperature']])
        
        # Save to DB
        engine = get_db_engine()
        forecast_df.to_sql('daily_forecasts', engine, if_exists='append', index=False)
        print("Success! Forecasts saved to database.")
        
    except Exception as e:
        print(f"Error generating forecast: {e}")
        print("Did you run train.py at least once?")

if __name__ == "__main__":
    generate_forecast()