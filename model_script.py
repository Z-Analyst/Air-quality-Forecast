# model_script.py
import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

# --- 1. CONFIGURATION ---
MONGO_URL = "mongodb://localhost:27017/"  # Placeholder - user must update
DATABASE_NAME = "air_quality_db"
COLLECTION_NAME = "nairobi" or "dar_es_salaam"
TARGET_COLUMN = "PM2.5"

# --- 2. DATA ACQUISITION & WRANGLING ---
def fetch_and_wrangle_data(uri, db_name, collection_name):
    """Fetches data from MongoDB and performs initial wrangling."""
    print("Connecting to MongoDB...")
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Example Query/Aggregation (Modify based on your original pipeline)
    pipeline = [
        # Example aggregation: sort, filter, or project fields
        {'$sort': {'timestamp': 1}},
    ]
    
    df = pd.DataFrame(list(collection.aggregate(pipeline)))
    
    # Time-Series Preprocessing
    df = df.set_index(pd.to_datetime(df['timestamp']))
    df = df.sort_index()
    # Handle missing values (e.g., ffill or interpolation)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(method='ffill') 

    return df[TARGET_COLUMN]

# --- 3. MODELING FUNCTIONS ---
def train_arima_model(series, order):
    """Trains an ARIMA model with a specified order."""
    model = ARIMA(series, order=order)
    results = model.fit()
    return results

def walk_forward_validation(data, train_size, order):
    """Performs walk-forward validation on the time series data."""
    train, test = data[:train_size], data[train_size:]
    history = [x for x in train]
    predictions = []
    
    for t in range(len(test)):
        # Train on history
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        # Predict the next step
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # Update history with the actual observation
        obs = test[t]
        history.append(obs)
        # print(f'Predicted={yhat}, Expected={obs}') # Optional: for verbose output

    # Evaluate performance
    mae = mean_absolute_error(test, predictions)
    print(f"Test MAE: {mae:.4f}")
    
    # Create the plot data
    results_df = pd.DataFrame({
        'Actual': test,
        'Predicted': predictions
    })
    return mae, results_df

# --- 4. VISUALIZATION ---
def plot_walk_forward_results(df):
    """Generates the key visualization showing actual vs. predicted values."""
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, dashes=False)
    plt.title('Walk Forward Validation: Actual vs. Predicted PM 2.5')
    plt.xlabel('Timestamp')
    plt.ylabel('PM 2.5 Reading')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('assets/walk_forward_prediction.png')
    plt.show()

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Data Retrieval
    air_quality_series = fetch_and_wrangle_data(MONGO_URI, DATABASE_NAME, COLLECTION_NAME)
    
    # 2. Define Best Model Order (Based on your tuning: e.g., (1, 0, 0) for AR(1))
    # NOTE: Replace with the actual best-performing order you found in the project!
    BEST_ARIMA_ORDER = (2, 0, 2) 
    
    # 3. Define Train/Test Split (e.g., 80% for training)
    TRAIN_SPLIT = int(len(air_quality_series) * 0.8)
    
    # 4. Perform Validation
    final_mae, plot_data = walk_forward_validation(
        data=air_quality_series, 
        train_size=TRAIN_SPLIT, 
        order=BEST_ARIMA_ORDER
    )
    
    # 5. Output Results and Plot
    print(f"\nFinal Test MAE (using Walk-Forward): {final_mae:.4f}")
    
    # Make sure the 'assets' folder exists before saving the plot
    import os
    if not os.path.exists('assets'):
        os.makedirs('assets')
        
    plot_walk_forward_results(plot_data)
