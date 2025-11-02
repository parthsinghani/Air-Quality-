import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_clean_data(file_name='AirQuality.csv'):
    """
    Implements Step 2 from the project plan.
    Loads, cleans, and preprocesses the AirQuality.csv file.
    """
    print(f"--- 1. Loading and Cleaning {file_name} ---")
    
    try:
        # 1. Load Data (delimiter is ';')
        df = pd.read_csv(file_name, sep=';')
    except FileNotFoundError:
        print(f"Error: '{file_name}' not found. Please make sure it's in the same directory.")
        return None

    # 2. Clean Corrupted Data
    # Drop empty rows at the end (read as NaNs)
    df.dropna(how='all', inplace=True)
    
    # Drop the last two unnamed/empty columns
    # We also drop NMHC(GT) as it's >90% missing values and not recoverable.
    try:
        df.drop(columns=[df.columns[-1], df.columns[-2], 'NMHC(GT)'], inplace=True)
    except KeyError as e:
        print(f"Warning: Could not drop column. {e}")

    # 5. Create Datetime Index
    # This must be done *before* type conversion
    try:
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
    except Exception as e:
        print(f"Error parsing datetime: {e}. Check Date/Time formats.")
        return None

    # 3 & 4. Handle Missing Values and Fix Data Types
    for col in df.columns:
        # First, fix the decimal separator (',' -> '.')
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        
        # Convert to numeric, forcing any errors (like blanks) to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Now, replace the -200 marker with NaN
        df[col] = df[col].replace(-200, np.nan)
        
    # Drop any columns that are *entirely* empty after cleaning
    df.dropna(axis=1, how='all', inplace=True)

    # 6. Impute Missing Data
    print(f"Total missing values before imputation: {df.isna().sum().sum()}")
    # Strategy: Time-based interpolation
    df.interpolate(method='time', inplace=True)
    
    # Fill any remaining NaNs (e.g., at the very start) with the next valid value
    df.fillna(method='bfill', inplace=True)
    
    print(f"Total missing values after imputation: {df.isna().sum().sum()}")
    print("--- Data Cleaning Complete ---")
    return df

def part1_anomaly_detection(df):
    """
    Implements Step 3 from the project plan.
    Uses IsolationForest to create an 'anomaly_score' feature.
    """
    print("\n--- Part 1: Running Anomaly Detection ---")
    
    # 3.1. Select Features
    # Use all columns except meteorological ones
    met_features = {'T', 'RH', 'AH'}
    anomaly_features = [col for col in df.columns if col not in met_features]
    print(f"Training anomaly model on {len(anomaly_features)} features.")

    # 3.2. Train Model
    # Contamination is the expected % of anomalies. 1% is a common baseline.
    iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(df[anomaly_features])

    # 3.3. Generate Scores
    # decision_function: lower scores are more anomalous
    df['anomaly_score'] = iso_forest.decision_function(df[anomaly_features])
    
    # 3.4. Visualize/Confirm Results (text-based)
    anomaly_count = (df['anomaly_score'] < 0).sum()
    print(f"Detected {anomaly_count} anomalous data points (approx. {anomaly_count/len(df)*100:.1f}%).")
    print("--- Part 1 Complete: 'anomaly_score' feature created ---")
    return df

def part2_classification_forecast(df):
    """
    Implements Step 4 & 5 from the project plan.
    Builds a classifier to predict the *next* hour's air quality.
    """
    print("\n--- Part 2: Building Classification Forecast Model ---")

    # 4.1. Create Target Variable (y)
    # Define classification thresholds for CO(GT)
    def classify_aq(co):
        if co >= 4.0:
            return 2  # "Unhealthy"
        elif co >= 2.0:
            return 1  # "Moderate"
        else:
            return 0  # "Good"

    df['AQ_Class'] = df['CO(GT)'].apply(classify_aq)
    
    # Crucial Step: Shift target to predict the *next* hour
    df['target'] = df['AQ_Class'].shift(-1)
    
    # 4.2. Create Feature Set (X)
    # Start with all available columns
    features = list(df.columns.drop(['AQ_Class', 'target']))
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    features.extend(['hour', 'day_of_week', 'month'])
    
    # Add lag features
    df['CO_lag_1hr'] = df['CO(GT)'].shift(1)
    df['CO_lag_3hr'] = df['CO(GT)'].shift(3)
    df['T_lag_1hr'] = df['T'].shift(1)
    features.extend(['CO_lag_1hr', 'CO_lag_3hr', 'T_lag_1hr'])

    # 4.3. Split Data (Time-Series Split)
    # Drop all NaNs created by shifting/lagging (at start and end of dataframe)
    df.dropna(inplace=True)
    
    X = df[features]
    y = df['target'].astype(int) # Ensure target is integer for classifier

    # DO NOT SHUFFLE. Split sequentially for time-series.
    split_percent = 0.8
    split_index = int(len(df) * split_percent)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target class distribution (Train): \n{y_train.value_counts(normalize=True)}")
    print(f"Target class distribution (Test): \n{y_test.value_counts(normalize=True)}")

    # 4.4. Train Classification Model
    # class_weight='balanced' helps the model pay more attention to rare "Unhealthy" class
    rf_classifier = RandomForestClassifier(n_estimators=100, 
                                         random_state=42, 
                                         class_weight='balanced', 
                                         n_jobs=-1)
    
    print("Training classification model...")
    rf_classifier.fit(X_train, y_train)
    print("--- Part 2 Model Training Complete ---")

    # 5. Evaluation & Interpretation
    print("\n--- 5. Model Evaluation ---")
    
    # 5.1. Generate Predictions
    y_pred = rf_classifier.predict(X_test)
    
    # 5.2. Check Performance
    target_names = ['0: Good', '1: Moderate', '2: Unhealthy']
    
    print("\nClassification Report (predicting *next* hour's class):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    print("Rows: Actual, Columns: Predicted")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), 
                       index=[f"Actual {c}" for c in target_names], 
                       columns=[f"Pred {c}" for c in target_names]))

    # 5.3. Interpret Results
    print("\n--- Feature Importances ---")
    feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    
    print(feature_importances.head(15))
    
    # Highlight our hybrid feature
    try:
        anomaly_rank = list(feature_importances.index).index('anomaly_score') + 1
        anomaly_imp = feature_importances['anomaly_score']
        print("\n------------------------------------------------------")
        print(f"HIGHLIGHT: 'anomaly_score' was the #{anomaly_rank} most important feature.")
        print(f"Importance Score: {anomaly_imp:.4f}")
        print("------------------------------------------------------")
    except ValueError:
        print("\n'anomaly_score' was not in the final feature set (this is unexpected).")

def main():
    """
    Main function to run the entire project pipeline.
    """
    # Step 2: Load and Clean
    df = load_and_clean_data(file_name='AirQuality.csv')
    
    if df is not None:
        # Step 3: Part 1 (Anomaly Detection)
        df_with_anomalies = part1_anomaly_detection(df)
        
        # Step 4 & 5: Part 2 (Classification)
        part2_classification_forecast(df_with_anomalies)

if __name__ == "__main__":
    main()
