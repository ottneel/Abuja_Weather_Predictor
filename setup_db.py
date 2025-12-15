import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 1. Force load the .env file
load_dotenv()

# 2. Get variables with safety checks
user = os.getenv('DB_USER', 'roots')
password = os.getenv('DB_PASS', 'roots')
host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '5432')
dbname = os.getenv('DB_NAME', 'abuja_air_quality')

# 3. Debug Print
print(f"DEBUG: Connecting as USER={user} on PORT={port} to DB={dbname}")

if port is None or port == 'None':
    print("CRITICAL ERROR: DB_PORT is still None. Hardcoding to 5432.")
    port = '5432'

# 4. Connect
try:
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_string)
    
    with engine.connect() as conn:
        print("Connected! Setting up tables...")

        # --- STEP A: Drop the old weak table ---
        conn.execute(text("DROP TABLE IF EXISTS temperature;"))
        print("Dropped old 'temperature' table.")

        # --- STEP B: Create the new SUPER table ---
        # (I fixed the syntax error here)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                timestamp TIMESTAMP,
                city VARCHAR(50),
                temperature FLOAT,
                humidity FLOAT,
                pm2_5 FLOAT,   -- For "P2" (Air Quality)
                pm10 FLOAT,    -- For "P1"
                source VARCHAR(20), -- To track if it came from "csv_history" or "api"
                PRIMARY KEY (timestamp, city)
            );
        """))
        print("Created 'sensor_data' table.")

        # --- STEP C: Create Forecast Table ---
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_forecasts (
                id SERIAL PRIMARY KEY,
                forecast_date DATE,
                predicted_temperature FLOAT,
                model_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        print("Created 'daily_forecasts' table.")

        conn.commit() # Save changes
        print(" SUCCESS: All database tables are ready!")

except Exception as e:
    print(f"\n CONNECTION FAILED: {e}")