import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 1. Force load the .env file
load_dotenv()

# 2. Get variables with safety checks
user = os.getenv('DB_USER', 'roots')      # Default to roots if missing
password = os.getenv('DB_PASS', 'roots')  # Default to roots if missing
host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '5432')       # Default to 5432 if missing
dbname = os.getenv('DB_NAME', 'abuja_air_quality')

# 3. Debug Print (Helps you see what is wrong)
print(f"DEBUG: Connecting as USER={user} on PORT={port} to DB={dbname}")

if port is None or port == 'None':
    print("CRITICAL ERROR: DB_PORT is still None. Hardcoding to 5432.")
    port = '5432'

# 4. Connect
try:
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_string)
    
    with engine.connect() as conn:
        # Create Tables
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS temperature (
                timestamp TIMESTAMP PRIMARY KEY,
                city VARCHAR(50),
                temperature FLOAT
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_forecasts (
                id SERIAL PRIMARY KEY,
                forecast_date DATE,
                predicted_temperature FLOAT,
                model_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit() # Important for some DB drivers
        print("✅ SUCCESS: Tables created successfully!")

except Exception as e:
    print(f"\n❌ CONNECTION FAILED: {e}")
    print("Check if your Docker Container is running: 'docker ps'")