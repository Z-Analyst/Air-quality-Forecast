a
# 🌍 Air Quality Prediction in Nairobi and Dar es Salaam

## 💡 Project Goal
Developed a time-series forecasting model to predict **PM 2.5 readings** (particulate matter) throughout the day for key stations in **Nairobi, Kenya, and Dar es Salaam, Tanzania**. The objective was to improve forecasting accuracy over a naive baseline, providing essential air quality insights for public health and environmental monitoring.

## 📊 Key Results
The final model significantly outperformed the naive baseline prediction.

* **Baseline Mean Absolute Error (MAE):** `4.053`
* **Final Model Mean Absolute Error (MAE):** `3.97`
    * *Result:* The model achieved an accuracy improvement of approximately **2.05%** over the simple persistence model, demonstrating effective time-series analysis and model selection.

## 🛠️ Methodology and Techniques

This project involved a comprehensive machine learning and data engineering workflow:

### 1. Data Wrangling & Preparation
* **Source:** Data was sourced from **openAfrica**.
* **Database:** Used **MongoDB** to efficiently store, query, and wrangle the time-series data. This demonstrated competency with NoSQL database management for large datasets.
* **Preprocessing:** Handled missing values, standardized timestamps, and aggregated readings to prepare the data for time-series modeling.

### 2. Time Series Modeling
* **Exploratory Data Analysis (EDA):** Used **Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots** to identify dependency in the time series data.
* **Models Explored:**
    * Linear Regression with Time-Series Features (as a benchmark)
    * **Autoregressive (AR) Models**
    * **Autoregressive Moving Average (ARMA) Models**
    * **ARIMA (Autoregressive Integrated Moving Average)**
* **Validation:** Employed **walk-forward validation**—a robust technique for time-series data—to simulate a real-world prediction environment and ensure the model's performance generalizes over time.

### 3. Hyperparameter Tuning
* Systematically tuned the orders of the AR, I, and MA components $(p, d, q)$ to minimize the Mean Absolute Error (MAE) on the validation set.

## 📈 Visual Insight

A key success of this project was the **walk-forward validation prediction**.  The visualization clearly showed the final model's predictions closely tracking the actual test data, confirming its efficacy in capturing the underlying temporal patterns of PM 2.5 fluctuations.

## ⚙️ Repository Structure

* `model_script.py`: Clean, modularized Python code containing all functions for data loading, modeling, and prediction.
* `requirements.txt`: A list of necessary libraries.
* `assets/`: Folder containing static images of key visualizations.

## 💻 Instructions to Run (Local Setup)

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data:** Due to WQU policy, the original data is not included. You will need to obtain the air quality data from **openAfrica** and configure your MongoDB connection settings within `model_script.py` (or a separate config file) to run the code.

   Initial Project README
