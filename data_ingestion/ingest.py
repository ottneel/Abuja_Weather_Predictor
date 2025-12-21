import os
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Abuja"
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

def fetch_and_store():
    print(f"Fetching weather data for {CITY}...")
    
    try:
        response = requests.get(URL)
        response.raise_for_status() # Stop if API request fails
        data = response.json()
        
        main_data = data['main']
        
        # Prepare the row for the new 'sensor_data' table
        row = {
            'timestamp': datetime.now(),
            'city': CITY,
            'temperature': main_data['temp'],
            'humidity': main_data['humidity'],
            'pm2_5': None,  # Standard API doesn't give this, so we leave it null for now
            'pm10': None,
            'source': 'api'
        }
        
        df = pd.DataFrame([row])
        
        # Save to DB
        engine = get_db_engine()
        df.to_sql('sensor_data', engine, if_exists='append', index=False)
        
        print(f"Success! Saved Temp: {row['temperature']}°C, Humidity: {row['humidity']}%")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_and_store()