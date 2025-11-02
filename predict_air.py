import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Set target variable
# We'll use 'C6H6(GT)' (Benzene) as the target, as it's a key pollutant.
TARGET_VARIABLE = 'C6H6(GT)'

print("--- ML Project Script Started ---")

# --- Step 1: Data Loading and Preprocessing (Revised) ---
print("\n--- Step 1: Loading and Preprocessing Data ---")
try:
    file_path = "D:\parth_ml\AirQuality.csv"
    # Load data, specifying the correct delimiter and decimal separator
    df = pd.read_csv(file_path, delimiter=';', decimal=',')
    
    print(f"Initial data loaded. Shape: {df.shape}")

    # Handle Missing Values (marked as -200)
    df.replace(-200, np.nan, inplace=True)
    
    # *** FIX ***
    # Drop rows where Date or Time is missing BEFORE creating the index
    # A row without a timestamp is not useful for time-series analysis
    df.dropna(subset=['Date', 'Time'], inplace=True)
    
    # Create a proper datetime index
    # The format %d/%m/%Y %H.%M.%S matches '10/03/2004 18.00.00'
    df['datetime_idx'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    
    # Check for any NaT (Not a Time) in the new index column, just in case
    if df['datetime_idx'].isnull().any():
        print("Warning: Found null values in datetime index. Dropping them.")
        df.dropna(subset=['datetime_idx'], inplace=True)
        
    df.set_index('datetime_idx', inplace=True)
    
    # Drop original Date/Time columns and any completely empty columns
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Drop 'NMHC(GT)' as it's mostly empty
    if 'NMHC(GT)' in df.columns:
        df.drop('NMHC(GT)', axis=1, inplace=True)
        
    print(f"Data shape after initial cleaning and index creation: {df.shape}")

    # Interpolate missing data using time-based method
    # This should now work as the index is clean
    df.interpolate(method='time', inplace=True)
    
    # Fill any remaining NaNs at the very beginning
    df.fillna(method='bfill', inplace=True)
    
    print("Data preprocessing complete. No missing values remaining.")
    print(df.info())

except Exception as e:
    print(f"Error during data loading/preprocessing: {e}")
    raise

# --- Step 2: Exploratory Data Analysis (EDA) ---
print("\n--- Step 2: Performing Exploratory Data Analysis (EDA) ---")
try:
    # Plot 1: Target variable over time
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[TARGET_VARIABLE], label=TARGET_VARIABLE, color='blue', alpha=0.8)
    plt.title(f'{TARGET_VARIABLE} (Benzene) Concentration Over Time')
    plt.xlabel('Date')
    plt.ylabel('Concentration (µg/m³)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('benzene_over_time.png')
    plt.clf()
    print("Generated 'benzene_over_time.png'")

    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Ensure all data is numeric for correlation
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='coolwarm', annot_kws={"size": 8})
    plt.title('Correlation Heatmap of All Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.clf()
    print("Generated 'correlation_heatmap.png'")

except Exception as e:
    print(f"Error during EDA: {e}")

# --- Step 3: Feature Engineering ---
print("\n--- Step 3: Performing Feature Engineering ---")
try:
    # Create Lag Features (past values of the target)
    df['target_lag_1hr'] = df[TARGET_VARIABLE].shift(1)
    df['target_lag_3hr'] = df[TARGET_VARIABLE].shift(3)
    df['target_lag_6hr'] = df[TARGET_VARIABLE].shift(6)
    
    # Create Time Features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Drop NaNs created by the shift() operation
    df.dropna(inplace=True)
    
    print("Feature engineering complete. Added lag and time features.")
    print(f"Final data shape for modeling: {df.shape}")

except Exception as e:
    print(f"Error during feature engineering: {e}")
    raise

# --- Step 4: Data Splitting (Time-Series) ---
print("\n--- Step 4: Splitting Data (Time-Series Split) ---")
try:
    # Define features (X) and target (y)
    y = df[TARGET_VARIABLE]
    X = df.drop(columns=[TARGET_VARIABLE])
    
    # CRITICAL: For time-series, DO NOT shuffle. Split chronologically.
    # Use first 80% for training, last 20% for testing
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Data split into training and testing sets:")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

except Exception as e:
    print(f"Error during data splitting: {e}")
    raise

# --- Step 5: Model Development and Evaluation ---
print("\n--- Step 5: Training and Evaluating Models ---")

# Initialize models as specified in the project PDF
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=5)
}

results = {}
predictions = {}

try:
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        preds = model.predict(X_test)
        predictions[name] = preds
        
        # Evaluate using metrics from the PDF
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        # Store results
        results[name] = {'RMSE': rmse, 'MAE': mae}
        
        print(f"Results for {name}:")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"  MAE (Mean Absolute Error):      {mae:.4f}")

    # Store the best model for feature importance (we'll use Random Forest)
    best_model_name = 'Random Forest'
    best_model = models[best_model_name]

except Exception as e:
    print(f"Error during model training/evaluation: {e}")

# --- Step 6: Visualization and Interpretation ---
print("\n--- Step 6: Visualizing Results ---")
try:
    # Plot 3: Prediction vs. Actual
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test, label='Actual Values', color='blue', alpha=0.7)
    plt.plot(y_test.index, predictions[best_model_name], label=f'{best_model_name} Predictions', color='red', linestyle='--')
    plt.title(f'Model Predictions vs. Actual Values ({best_model_name})')
    plt.xlabel('Date')
    plt.ylabel('Concentration (µg/m³)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_vs_actual.png')
    plt.clf()
    print("Generated 'prediction_vs_actual.png'")

    # Plot 4: Feature Importance
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    # Plot top 20 features
    top_n = 20
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.clf()
    print("Generated 'feature_importance.png'")

except Exception as e:
    print(f"Error during result visualization: {e}")

# --- Final Summary ---
print("\n--- ML Project Script Finished ---")
print("\nFinal Model Metrics:")
results_df = pd.DataFrame(results).T
print(results_df)

print("\nScript complete. All plots and evaluations have been generated.")