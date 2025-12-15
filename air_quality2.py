#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import glob
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
from time import time
from typing import Dict
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings
from typing import Union
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json
import tempfile
import os
warnings.filterwarnings('ignore')


# set tracking point for mlflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")


#mlflow.set_tracking_uri("postgresql://postgres:roots@localhost:5432/abuja_air_quality_data")



#function to read the csv
def read_sensor_data(folder_path):
    headers = [
        'sensor_id', 
        'sensor_type', 
        'location', 
        'lat', 
        'lon', 
        'timestamp', 
        'value_type', 
        'value'
    ]

    csv_files = glob.glob(folder_path + '/*csv')
    df= pd.concat((pd.read_csv(file, header=None, names=["combined_data"]) for file in csv_files), ignore_index= True)
    
    df[headers] = df['combined_data'].str.split(';', expand=True)
    df = df.drop(columns=['combined_data']).iloc[1:].reset_index(drop=True)
    return df



df = read_sensor_data(r"C:\Users\hpuser\air_quality_data")



#seperate the data according to value_type
def separate_data(df):
    # Define the value types to separate
    value_types = ["P2", "P1", "P0", "humidity", "temperature"]
    
    # Use a dictionary to store the separated dataframes
    separated_data = {value_type: df.loc[df["value_type"] == value_type].copy() 
                      for value_type in value_types}
    
    # Return the separated dataframes
    return separated_data



separated_data = separate_data(df)


# In[ ]:


#renaming the value column and droping the value_type column
def rename_value_column(dataframe):
    new_column_name = dataframe["value_type"].iloc[0]
    dataframe.rename(columns={"value": new_column_name}, inplace=True)
    dataframe.drop(columns=['value_type'], inplace=True)
    return dataframe


# In[ ]:


for value_type, dataframe in separated_data.items():
    separated_data[value_type] = rename_value_column(dataframe)


# In[ ]:


# Function to create PostgreSQL database
def create_postgres_db ( db_name: str, username: str, password: str, host: str = 'localhost', port: int = 5432) -> None:
    """
    Create a new PostgreSQL database
    
    Args:
        db_name: Name of the database to create
        username: PostgreSQL username
        password: PostgreSQL password
        host: Database host (default: localhost)
        port: Database port (default: 5432)
    """
    # Connect to default 'postgres' database to create a new database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        dbname="postgres"  # Connect to default postgres database
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Check if database already exists
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
    exists = cursor.fetchone()
    
    if not exists:
        try:
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully")
        except Exception as e:
            print(f"Error creating database: {e}")
    else:
        print(f"Database '{db_name}' already exists")
    
    cursor.close()
    conn.close()


# In[ ]:


# Function to load DataFrames into PostgreSQL
def load_dataframes_to_postgres(
    dataframes: Dict[str, pd.DataFrame], 
    db_name: str, 
    username: str, 
    password: str, 
    host: str = 'localhost', 
    port: int = 5432,
    if_exists: str = 'replace'
) -> None:
    """
    Load multiple pandas DataFrames into PostgreSQL tables
    
    Args:
        dataframes: Dictionary mapping table names to pandas DataFrames
        db_name: Name of the target database
        username: PostgreSQL username
        password: PostgreSQL password
        host: Database host (default: localhost)
        port: Database port (default: 5432)
        if_exists: How to behave if the table exists ('fail', 'replace', 'append') (default: 'replace')
    """
    # Create SQLAlchemy engine
    engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{db_name}')
    
    # Load each DataFrame into a table
    for table_name, df in dataframes.items():
        try:
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,
                index=False,  # Don't store DataFrame indices
                chunksize=1000  # Load data in chunks
            )
            print(f"Successfully loaded DataFrame to table '{table_name}'")
        except Exception as e:
            print(f"Error loading DataFrame to table '{table_name}': {e}")


# In[ ]:


if __name__ == "__main__":
    # Database credentials
    DB_NAME = "abuja_air_quality_data"
    USERNAME = "postgres"
    PASSWORD = "roots"
    HOST = "localhost"
    PORT = 5432
    
    # Create database and load DataFrames
    create_postgres_db(DB_NAME, USERNAME, PASSWORD, HOST, PORT)
    load_dataframes_to_postgres(separated_data, DB_NAME, USERNAME, PASSWORD, HOST, PORT)


# In[5]:


DB_NAME = "abuja_air_quality_data"
USERNAME = "postgres"
PASSWORD = "roots"
HOST = "localhost"
PORT = 5432

# Testing the connection and querying data
engine = create_engine(f'postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}')
df = pd.read_sql("SELECT timestamp, temperature FROM temperature order by timestamp asc;", engine)


# In[6]:


def timestamp_preprocess(df):
    # Make the timestamp column the Index
    df.set_index('timestamp', inplace =True)
    # Turn the datatype of the index column from base to Datetime
    df.index= pd.to_datetime(df.index)
    # Turn the Time zone to that of Nigeria
    df.index = df.index.tz_convert("Africa/Lagos")
    return df


# In[7]:


df = timestamp_preprocess(df)


# In[8]:


def temperature_preprocessing(df):
    #convert the temperature column to float from object and assign invalid inputations as NaN
    df["temperature"] = pd.to_numeric(df["temperature"], errors='coerce')
    
    # Df mask to remove outliers
    df = df[(df['temperature'] >= 15.0) & (df['temperature'] <= 43.0)]
    
    # Get the Mean Temperature of each day
    df = df["temperature"].resample("1D").mean().to_frame()

    return df


# In[9]:


df = temperature_preprocessing(df)


# In[10]:


#BoxPlot to check the distrubition of the points in temperature column to confirm Outlier treatment
fig,ax = plt.subplots(figsize =(15,6));
df["temperature"].plot(kind="box", vert=False, title="Distrubution of Temperature Data", ax=ax);


# In[11]:


# Calculate and plot the weekly rolling average temperature
fig, ax = plt.subplots(figsize=(15, 6))
df["temperature"].rolling(7).mean().plot(ax=ax, ylabel="Temperature", title="7-day Rolling Average Temperature");


# In[12]:


# We see gaps in our datapoint which means there are missing values that will need to be treated.
# we also confirm with the Info method
df.info()


# In[13]:


def apply_seasonal_adjustment(df):
    """
    Apply seasonal adjustment to fill missing temperature data.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with datetime index and 'temperature' column
    
    Returns:
    pandas.DataFrame: DataFrame with filled values
    """

    # Create a copy of the dataframe
    filled_df = df.copy()
    
    # First, fill the missing days in 2023 since we need complete 2023 data
    # to properly apply seasonal adjustment for 2024
    
    # Fill scattered missing days in 2023 (January through December)
    for month in range(1, 13):
        month_missing = df[(df.index.year == 2023) & (df.index.month == month) & df['temperature'].isna()]
        if not month_missing.empty:
            month_avg = df[(df.index.year == 2023) & (df.index.month == month) & ~df['temperature'].isna()]['temperature'].mean()
            
            for missing_date in month_missing.index:
                filled_df.loc[missing_date, 'temperature'] = month_avg
                month_name = missing_date.strftime('%B')
                print(f"Filled {missing_date.date()} with {month_name} 2023 average: {month_avg:.2f}°C")
    
    # Now that 2023 is complete, calculate adjustment factor for 2024
    # Use available complete months from both years for comparison
    comparison_months = []
    
    # Check which months have data in both years
    for month in range(1, 13):
        complete_2023 = not filled_df[(filled_df.index.year == 2023) & (filled_df.index.month == month)].empty
        complete_2024 = not filled_df[(filled_df.index.year == 2024) & (filled_df.index.month == month) & ~filled_df['temperature'].isna()].empty
        
        if complete_2023 and complete_2024:
            comparison_months.append(month)
    
    print(f"Using months for comparison: {[pd.Timestamp(2023, m, 1).strftime('%B') for m in comparison_months]}")
    
    monthly_diffs = []
    for month in comparison_months:
        # Get data for 2023 and 2024
        temp_2023 = filled_df[(filled_df.index.year == 2023) & (filled_df.index.month == month)]['temperature'].mean()
        temp_2024 = filled_df[(filled_df.index.year == 2024) & (filled_df.index.month == month) & ~filled_df['temperature'].isna()]['temperature'].mean()
        
        # Calculate difference
        diff = temp_2024 - temp_2023
        monthly_diffs.append(diff)
        month_name = pd.Timestamp(2023, month, 1).strftime('%B')
        print(f"{month_name} difference: {diff:.3f}°C")
    
    # Calculate the adjustment factor
    adjustment_factor = sum(monthly_diffs) / len(monthly_diffs)
    print(f"\nCalculated adjustment factor: {adjustment_factor:.3f}°C")
    
    # Fill the missing data for 2024 using corresponding 2023 values + adjustment
    for month in range(1, 13):  # All months
        # Get all days in this month in 2024 that are missing data
        missing_days_2024 = filled_df[(filled_df.index.year == 2024) & 
                                      (filled_df.index.month == month) & 
                                      filled_df['temperature'].isna()]
        
        # For each missing day
        for missing_date in missing_days_2024.index:
            # Find corresponding day in 2023
            try:
                date_2023 = pd.Timestamp(year=2023, month=missing_date.month, day=missing_date.day)
                
                # If 2023 data exists, use it with adjustment
                if date_2023 in filled_df.index:
                    filled_value = filled_df.loc[date_2023, 'temperature'] + adjustment_factor
                    filled_df.loc[missing_date, 'temperature'] = filled_value
                    print(f"Filled {missing_date.date()} with adjusted 2023 value: {filled_value:.2f}°C")
            except:
                # Skip invalid dates (like Feb 29 in non-leap years)
                continue
    
    # Check if there are still missing values and handle them
    still_missing = filled_df[filled_df['temperature'].isna()]
    if not still_missing.empty:
        print(f"\nStill have {len(still_missing)} missing values. Filling with monthly averages...")
        
        for missing_date in still_missing.index:
            # Use monthly average from 2023 + adjustment
            month_avg_2023 = filled_df[(filled_df.index.year == 2023) & 
                                      (filled_df.index.month == missing_date.month)]['temperature'].mean()
            
            filled_value = month_avg_2023 + adjustment_factor
            filled_df.loc[missing_date, 'temperature'] = filled_value
            month_name = missing_date.strftime('%B')
            print(f"Filled {missing_date.date()} with adjusted {month_name} 2023 average: {filled_value:.2f}°C")
    
    # Check how many values are still missing
    missing_before = df['temperature'].isna().sum()
    missing_after = filled_df['temperature'].isna().sum()
    print(f"\nMissing values before: {missing_before}")
    print(f"Missing values after: {missing_after}")
    print(f"Filled {missing_before - missing_after} values")
    
    return filled_df


# In[14]:


# Apply the function
df = apply_seasonal_adjustment(df)


# In[15]:


#confirm if we have any missing values,
df.isnull().sum()


# In[16]:


#Visualize the changes
fig, ax = plt.subplots(figsize=(15, 6))
df["temperature"].rolling(7).mean().plot(ax= ax, ylabel="Temperature", title="7-day Rolling Average");


# In[18]:


# Custom wrapper for ARIMA model
class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, arima_model):
        self.arima_model = arima_model
    
    def predict(self, context, model_input):
        # model_input should contain number of steps to forecast
        steps = model_input.iloc[0, 0] if hasattr(model_input, 'iloc') else model_input
        return self.arima_model.forecast(steps=steps)


# In[19]:


mlflow.set_experiment("Abuja_Temperature_Pred")


# In[20]:


with mlflow.start_run(run_name = "time_series_diagnostics"):
    mlflow.set_tag("components", "stationarity,autocorrelation,partial_autocorrelation") 
    
    # ADF Test - using temperature column for consistency
    Adfuller_testing = adfuller(df['temperature'])
    Adfuller_statistics = Adfuller_testing[0]
    Pvalue = Adfuller_testing[1]
    mlflow.log_param("p-value", Pvalue)
    mlflow.log_param("Adfuller_statistics", Adfuller_statistics)
    
    # 2. ACF PLOT WITH EXTENDED LAGS (to capture yearly seasonality)
    plt.figure(figsize=(14, 7))
    plot_acf(df['temperature'], lags=800, alpha=0.05)
    plt.title('Autocorrelation Function (ACF) - Full Range')
    plt.grid(True)
    plt.tight_layout()
    acf_filename = "ACF_Plot.jpg"
    plt.savefig(acf_filename)
    mlflow.log_artifact(acf_filename)
    plt.close()  # Close figure to free memory
    
    # 4. PARTIAL AUTOCORRELATION (PACF)
    plt.figure(figsize=(15, 9))
    plot_pacf(df['temperature'], lags=50, alpha=0.05, method='ywm')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.grid(True)
    plt.tight_layout()
    pacf_filename = "PACF_Plot.jpg"
    plt.savefig(pacf_filename)
    mlflow.log_artifact(pacf_filename)
    plt.close()  # Close figure to free memory


# In[21]:


def train_test_spliting(df):
    # Calculate the index where the last 30% begins
    split_idx = int(len(df) * 0.7)
    
    # Select the last 30% as test data
    test_data = df[split_idx:]
    
    # The first 70% would be your training data
    train_data = df[:split_idx]

    return train_data, test_data


# In[22]:


train_data, test_data = train_test_spliting(df)


# In[23]:


#ARIMA MODEL RUN
def train_model(train_data):
    with mlflow.start_run(run_name="ARIMA Model"):
        
        # Define the p, d, q parameters for the grid search
        p_range = range(0, 3)  # 0, 1, 2
        d_range = range(0, 2)  # 0, 1
        q_range = range(0, 3)  # 0, 1, 2
        
        # Create all combinations of p, d, q
        pdq = list(itertools.product(p_range, d_range, q_range))
        
        # Store Params in Mlflow
        mlflow.log_param("grid_search", json.dumps(pdq))
        
        # Create a DataFrame to store results
        results = pd.DataFrame(columns=['pdq', 'AIC', 'BIC'])
        
        # Run the grid search
        print("Running ARIMA grid search...")
        for param in pdq:
            try:
                # Create and fit the ARIMA model
                model = ARIMA(train_data, order=param)
                model_fit = model.fit()
                
                # Save the results
                results = pd.concat([results, pd.DataFrame({
                    'pdq': [param],
                    'AIC': [model_fit.aic],
                    'BIC': [model_fit.bic]
                })], ignore_index=True)
                
            except Exception as e:
                print(f'ARIMA{param} - Error: {e}')
                continue
        
        # Log the results as CSV artifact
        results.to_csv("grid_search_results.csv", index=False)
        mlflow.log_artifact("grid_search_results.csv")
        
        # Sort results by AIC
        results_sorted = results.sort_values(by='AIC')
        top3_AIC = results_sorted.head(3)
        
        # Log Best 3 by AIC as CSV artifact
        top3_AIC.to_csv("top3_AIC.csv", index=False)
        mlflow.log_artifact("top3_AIC.csv")
        
        # Sort results by BIC
        results_sorted_bic = results.sort_values(by='BIC')
        top3_BIC = results_sorted_bic.head(3)
        
        # Log Best 3 by BIC as CSV artifact
        top3_BIC.to_csv("top3_BIC.csv", index=False)
        mlflow.log_artifact("top3_BIC.csv")
        
        # Get the best model parameters
        best_aic_params = results_sorted.iloc[0]
        mlflow.log_param("best_param", str(best_aic_params['pdq']))
        mlflow.log_metric("best_aic", best_aic_params['AIC'])
        mlflow.log_metric("best_bic", best_aic_params['BIC'])
        
        # Fit the best model
        start = time()
        best_order = best_aic_params['pdq']
        best_model = ARIMA(train_data, order=best_order)
        best_model_fit = best_model.fit()
        end = time()
        time_taken = end - start
        mlflow.log_metric("model_fitting_time", time_taken)
        
        # Log the best model summary
        summary = str(best_model_fit.summary())
        with open("model_summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("model_summary.txt")

        # Log the model
        wrapped_model = ARIMAModelWrapper(best_model_fit)
        mlflow.pyfunc.log_model(
            artifact_path="logged_arima_model",
            python_model=wrapped_model,
            conda_env={
                'channels': ['defaults'],
                'dependencies': [
                    'python=3.8',
                    'statsmodels',
                    'pandas',
                    'numpy'
                ]
            }
        )
        
        # STANDARD EVALUATION
        print("Running standard evaluation...")

        start_standard = time()
        # Standard forecast - fit once, predict all test periods
        n_test_periods = len(test_data)
        standard_forecasts = best_model_fit.forecast(steps=n_test_periods)

        end_standard = time()
        standard_time = end_standard - start_standard

        mlflow.log_metric("standard_evaluation_time", standard_time)
        
        # Calculate standard metrics
        standard_rmse = np.sqrt(mean_squared_error(test_data, standard_forecasts))
        standard_mae = mean_absolute_error(test_data, standard_forecasts)
        
        # Log standard metrics
        mlflow.log_metric("standard_rmse", standard_rmse)
        mlflow.log_metric("standard_mae", standard_mae)
        
        print(f"Standard RMSE: {standard_rmse:.4f}")
        print(f"Standard MAE: {standard_mae:.4f}")
        
        # ROLLING WINDOW EVALUATION
        print("Running rolling window evaluation...")
        
        rolling_predictions = pd.Series(index=test_data.index, dtype=float)
        
        start_rolling = time()
        for end_date in test_data.index:
            # Create training data up to the day before the prediction date
            rolling_train_data = df.loc[:end_date - timedelta(days=1)]
            
            # Fit ARIMA model on rolling window
            rolling_model = ARIMA(rolling_train_data, order=best_order)
            rolling_model_fit = rolling_model.fit()
            
            # Generate one-step forecast for the end_date
            forecast = rolling_model_fit.forecast(steps=1)
            
            # Store the prediction
            rolling_predictions.loc[end_date] = forecast[0]
        
        end_rolling = time()
        rolling_time = end_rolling - start_rolling
        
        # Calculate rolling window metrics
        rolling_rmse = np.sqrt(mean_squared_error(test_data, rolling_predictions))
        rolling_mae = mean_absolute_error(test_data, rolling_predictions)
        
        # Log rolling window metrics
        mlflow.log_metric("rolling_rmse", rolling_rmse)
        mlflow.log_metric("rolling_mae", rolling_mae)
        mlflow.log_metric("rolling_evaluation_time", rolling_time)
        
        # Save predictions as artifacts
        standard_results = pd.DataFrame({
            'actual': test_data.squeeze(),
            'predicted': standard_forecasts.squeeze()
        }, index=test_data.index if hasattr(test_data, 'index') else None)
        standard_results.to_csv("standard_predictions.csv")
        mlflow.log_artifact("standard_predictions.csv")
        
        rolling_results = pd.DataFrame({
            'actual': test_data.squeeze(),
            'predicted': rolling_predictions.squeeze()
        }, index=test_data.index if hasattr(test_data, 'index') else None)
        rolling_results.to_csv("rolling_predictions.csv")
        mlflow.log_artifact("rolling_predictions.csv")
        
        # Compare approaches
        print(f"\nComparison:")
        print(f"Standard vs Rolling RMSE: {standard_rmse:.4f} vs {rolling_rmse:.4f}")
        print(f"Standard vs Rolling MAE: {standard_mae:.4f} vs {rolling_mae:.4f}")
        
        # Log comparison metrics
        mlflow.log_metric("rmse_difference", abs(standard_rmse - rolling_rmse))
        mlflow.log_metric("mae_difference", abs(standard_mae - rolling_mae))


# In[24]:


with mlflow.start_run(run_name="Forecast Comparison Charts"):
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.squeeze(), label='Actual')
    plt.plot(rolling_predictions, label='Predicted', color='red')
    plt.title('ARIMA Model: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    rolling_forecast_comparison = "Rolling_Forecast_Comparison.png"
    plt.savefig(rolling_forecast_comparison)
    mlflow.log_artifact(rolling_forecast_comparison, artifact_path = "comparison_plots")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(test_data.squeeze(), label='Actual')
    plt.plot(standard_forecasts, label='Standard_Predicted', color='brown')
    plt.title('ARIMA Model: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    standard_forecast_comparison = "Standard_Forecast_Comparison.png"
    plt.savefig(standard_forecast_comparison)
    mlflow.log_artifact(standard_forecast_comparison, artifact_path = "comparison_plots")
    plt.close()


# In[25]:


with mlflow.start_run(run_name="arima_residuals_analysis"):
    # Plot residuals for Standard forecasts
    stand_residuals = test_data.squeeze() - standard_forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(stand_residuals)
    plt.title('Residuals of Standard ARIMA Model')
    plt.tight_layout()
    standard_residuals = 'arima_residuals.png'
    plt.savefig(standard_residuals)
    mlflow.log_artifact(standard_residuals, artifact_path="residual_plots")
    plt.close()
    
    # Plot residuals for Rolling Predictions
    roll_residuals = test_data.squeeze() - rolling_predictions
    plt.figure(figsize=(12, 6))
    plt.plot(roll_residuals)
    plt.title('Residuals of Rolling Predictions ARIMA Model')
    plt.tight_layout()
    rolling_residuals = 'rolling_residuals.png'
    plt.savefig(rolling_residuals)
    mlflow.log_artifact(rolling_residuals, artifact_path="residual_plots")
    plt.close()

    plot_acf(roll_residuals, lags=40, alpha=0.05, title='ACF of rolling_Residuals')
    plt.tight_layout()
    acf_rolling_residuals = "acf_rolling_residuals.png"
    plt.savefig(acf_rolling_residuals)
    mlflow.log_artifact(acf_rolling_residuals, artifact_path = "residual_plots")
    plt.close()

    plot_acf(stand_residuals, lags=40, alpha=0.05, title='ACF of Standard_Residuals')
    plt.tight_layout()
    acf_standard_residuals = "acf_standard_residuals.png"
    plt.savefig(acf_standard_residuals)
    mlflow.log_artifact(acf_standard_residuals, artifact_path = "residual_plots")
    plt.close()


# In[26]:


# Sarima Model Run
with mlflow.start_run(run_name="Sarima Model Test I"):
    p_range = range(0, 3)  # 0, 1, 2
    d = 1
    q_range = range(0, 3)  # 0, 1, 2
    P_range = range(0, 2)  # 0, 1
    D = 1
    Q_range = range(0, 2)  # 0, 1
    s = 91
    
    # Create all combinations of p, q, P, Q
    pdq = list(itertools.product(p_range, [d], q_range))
    seasonal_pdq = list(itertools.product(P_range, [D], Q_range, [s]))
    mlflow.log_param("Sarima_Grid_Search", json.dumps(seasonal_pdq))
    
    # Create a DataFrame to store results
    results = pd.DataFrame(columns=['pdq', 'seasonal_pdq', 'AIC', 'BIC'])
    
    # Run the grid search
    print("Running SARIMA grid search...")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                # Create and fit the SARIMA model
                model = SARIMAX(train_data,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                    
                model_fit = model.fit(disp=False)
                
                # Save the results
                results = pd.concat([results, pd.DataFrame({
                    'pdq': [param],
                    'seasonal_pdq': [param_seasonal],
                    'AIC': [model_fit.aic],
                    'BIC': [model_fit.bic]
                })], ignore_index=True)
     
            except Exception as e:
                print(f'SARIMA{param}x{param_seasonal} - Error: {e}')
                continue
    
    # Save and log results
    results.to_csv("sarima_results.csv", index=False)
    mlflow.log_artifact("sarima_results.csv", artifact_path="sarima_data")
    
    # Sort results by AIC
    results_sorted = results.sort_values(by='AIC')
    best_metric = results_sorted.iloc[0]
    
    # Log best model metrics and parameters
    mlflow.log_metric("best_AIC", best_metric['AIC'])
    mlflow.log_metric("best_BIC", best_metric['BIC'])
    mlflow.log_param("best_pdq", str(best_metric['pdq']))
    mlflow.log_param("best_seasonal_pdq", str (best_metric['seasonal_pdq']))


# # Second MLflow run for model evaluation
# with mlflow.start_run(run_name="Sarima Model test running the best model"):
#     # Fit the best model
#     start = time()
#     best_model = SARIMAX(train_data,
#                 order=best_metric["pdq"],
#                 seasonal_order=best_metric["seasonal_pdq"],
#                 enforce_stationarity=True,
#                 enforce_invertibility=True)
# 
#     best_fit = best_model.fit(disp=False)
#     end = time()
#     time_taken = end - start
#     mlflow.log_metric("model_fitting_time", time_taken)
#     
#     # Log the best model summary
#     summary = str(best_fit.summary())
#     summary_file = "model_summary.txt"
#     try:
#         with open(summary_file, "w") as f:
#             f.write(summary)
#         mlflow.log_artifact(summary_file, artifact_path="model_artifacts")
#     finally:
#         # Clean up temporary file
#         if os.path.exists(summary_file):
#             os.remove(summary_file)
#     
#     # Log the model
#     wrapped_sarima_model = ARIMAModelWrapper(best_fit)
#     mlflow.pyfunc.log_model(
#         artifact_path="logged_sarima_model",
#         python_model=wrapped_sarima_model,
#         conda_env={
#             'channels': ['defaults'],
#             'dependencies': [
#                 'python=3.8',
#                 'statsmodels',
#                 'pandas',
#                 'numpy'
#             ]
#         }
#     )
#     
#     # STANDARD EVALUATION
#     print("Running standard evaluation...")
# 
#     start_standard = time()
#     # Standard forecast - fit once, predict all test periods
#     n_test_periods = len(test_data)
#     standard_forecasts = best_fit.forecast(steps=n_test_periods)
# 
#     end_standard = time()
#     standard_time = end_standard - start_standard
# 
#     mlflow.log_metric("standard_evaluation_time", standard_time)
#     
#     # Calculate standard metrics
#     standard_rmse = np.sqrt(mean_squared_error(test_data, standard_forecasts))
#     standard_mae = mean_absolute_error(test_data, standard_forecasts)
#     
#     # Log standard metrics
#     mlflow.log_metric("standard_rmse", standard_rmse)
#     mlflow.log_metric("standard_mae", standard_mae)
#     
#     print(f"Standard RMSE: {standard_rmse:.4f}")
#     print(f"Standard MAE: {standard_mae:.4f}")
#     
#     # ROLLING WINDOW EVALUATION
#     print("Running rolling window evaluation...")
#     
#     rolling_predictions = pd.Series(index=test_data.index, dtype=float)
#     
#     start_rolling = time()
#     for end_date in test_data.index:
#         try:
#             # Create training data up to the day before the prediction date
#             rolling_train_data = train_data.loc[:end_date - timedelta(days=1)]
#             
#             # Fit SARIMA model on rolling window - FIXED: using best_metric instead of best_aic_params
#             rolling_model = SARIMAX(rolling_train_data, 
#                                    order=best_metric['pdq'],
#                                    seasonal_order=best_metric['seasonal_pdq'],
#                                    enforce_stationarity=True,
#                                    enforce_invertibility=True)
#             rolling_model_fit = rolling_model.fit(disp=False)
#             
#             # Generate one-step forecast for the end_date
#             forecast = rolling_model_fit.forecast(steps=1)
#             
#             # Store the prediction
#             rolling_predictions.loc[end_date] = forecast[0]
#             
#         except Exception as e:
#             print(f"Error in rolling prediction for {end_date}: {e}")
#             continue
#     
#     end_rolling = time()
#     rolling_time = end_rolling - start_rolling
#     
#     # Calculate rolling window metrics
#     rolling_rmse = np.sqrt(mean_squared_error(test_data, rolling_predictions))
#     rolling_mae = mean_absolute_error(test_data, rolling_predictions)
#     
#     # Log rolling window metrics
#     mlflow.log_metric("rolling_rmse", rolling_rmse)
#     mlflow.log_metric("rolling_mae", rolling_mae)
#     mlflow.log_metric("rolling_evaluation_time", rolling_time)
#     
#     print(f"Rolling RMSE: {rolling_rmse:.4f}")
#     print(f"Rolling MAE: {rolling_mae:.4f}")
#     print(f"Rolling evaluation time: {rolling_time:.2f} seconds")
#     
#     # Save predictions as artifacts
#     standard_results = pd.DataFrame({
#         'actual': test_data.squeeze(),
#         'predicted': standard_forecasts.squeeze()
#     }, index=test_data.index if hasattr(test_data, 'index') else None)
#     
#     rolling_results = pd.DataFrame({
#         'actual': test_data.squeeze(),
#         'predicted': rolling_predictions.squeeze()
#     }, index=test_data.index if hasattr(test_data, 'index') else None)
#     
#     # Save and log prediction files with cleanup
#     standard_file = "standard_predictions.csv"
#     rolling_file = "rolling_predictions.csv"
#     
#     try:
#         standard_results.to_csv(standard_file)
#         mlflow.log_artifact(standard_file, artifact_path="predictions")
#         
#         rolling_results.to_csv(rolling_file)
#         mlflow.log_artifact(rolling_file, artifact_path="predictions")
#         
#     finally:
#         # Clean up temporary files
#         for file_path in [standard_file, rolling_file]:
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#     
#     # Compare approaches
#     print(f"\nComparison:")
#     print(f"Standard vs Rolling RMSE: {standard_rmse:.4f} vs {rolling_rmse:.4f}")
#     print(f"Standard vs Rolling MAE: {standard_mae:.4f} vs {rolling_mae:.4f}")
#     
#     # Log comparison metrics
#     mlflow.log_metric("rmse_difference", abs(standard_rmse - rolling_rmse))
#     mlflow.log_metric("mae_difference", abs(standard_mae - rolling_mae))

# In[28]:


# To forecast just the next day
next_day_forecast = best_model_fit.forecast(steps=1)[0]
next_date = test_data.index[-1] + pd.Timedelta(days=1)
print(f"Forecast for {next_date.date()}: {next_day_forecast:.2f}°C")


# In[ ]:




