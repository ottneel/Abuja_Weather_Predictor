import os
import glob
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

def clean_series(series, name):
    """
    Applies physics-based limits to sensor data.
    """
    series = pd.to_numeric(series, errors='coerce')
    
    if name == 'temperature':
        # Abuja record low is ~15C, High is ~40C. Buffer 10-50.
        series = series.mask((series < 10) | (series > 50))
    elif name == 'humidity':
        series = series.mask((series < 0) | (series > 100))
    elif name in ['pm2_5', 'pm10']:
        series = series.mask(series < 0) 
    
    return series

def process_csv_history():
    # folder path
    raw_path = "./we_csv_files" 
    
    # Check if folder exists
    if not os.path.exists(raw_path):
        print(f" ERROR: Folder '{raw_path}' not found!")
        print("Make sure you created this folder and put your CSVs inside it.")
        return

    csv_files = glob.glob(os.path.join(raw_path, "*.csv"))
    
    if not csv_files:
        print(f" No CSV files found in '{raw_path}'.")
        return

    print(f"Found {len(csv_files)} files. Reading and parsing...")
    
    all_data = []
    
    # 1. Read the data 
    for file in csv_files:
        try:
            # Read file without header
            temp_df = pd.read_csv(file, header=None, names=["combined_data"])
            
            # Split the semicolon format
            split_data = temp_df['combined_data'].str.split(';', expand=True)
            
            # Extract relevant columns (Index 5=Time, 6=Type, 7=Value)
            clean_df = pd.DataFrame({
                'timestamp': pd.to_datetime(split_data[5], errors='coerce'),
                'value_type': split_data[6],
                'value': split_data[7]
            })
            
            clean_df = clean_df.dropna(subset=['timestamp'])
            all_data.append(clean_df)
            
        except Exception as e:
            print(f"Skipping corrupt file {file}: {e}")

    if not all_data:
        print("No valid data extracted.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # 2. Restructure the Dataset
    print("Pivoting data...")
    metrics_map = {
        'temperature': 'temperature',
        'humidity': 'humidity',
        'P2': 'pm2_5',
        'P1': 'pm10'
    }
    
    full_df = full_df[full_df['value_type'].isin(metrics_map.keys())]
    full_df['value'] = pd.to_numeric(full_df['value'], errors='coerce')
    
    pivot_df = full_df.pivot_table(index='timestamp', columns='value_type', values='value', aggfunc='mean')
    pivot_df = pivot_df.rename(columns=metrics_map)
    
    for col in metrics_map.values():
        if col not in pivot_df.columns:
            pivot_df[col] = np.nan

    # 3. Resampling the data and filling missing data
    print("Cleaning and Resampling...")
    
    daily_df = pivot_df.resample('D').mean()
    
    for col in daily_df.columns:
        daily_df[col] = clean_series(daily_df[col], col)

    # Imputation
    daily_df = daily_df.interpolate(method='time', limit=3)
    daily_df = daily_df.ffill().bfill()

    daily_df['city'] = 'Abuja'
    daily_df['source'] = 'csv_history'
    daily_df = daily_df.reset_index()

    # 4. Uploading the data
    print(f"Uploading {len(daily_df)} clean daily records to DB...")
    engine = get_db_engine()
    
    try:
        daily_df.to_sql('sensor_data', engine, if_exists='append', index=False)
        print(" Success! History loaded.")
        print(daily_df.head())
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    process_csv_history()