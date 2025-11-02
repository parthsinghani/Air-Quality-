import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Air Quality Report")
warnings.filterwarnings('ignore')

# --- 1. Data Loading and Cleaning ---
# This function is cached to run only once
@st.cache_data
def load_and_clean_data(file_name='AirQuality.csv'):
    """
    Loads, cleans, and preprocesses the AirQuality.csv file.
    """
    st.write("Cache: Running Data Cleaning...")
    try:
        # Load the CSV file with the correct semicolon delimiter
        df = pd.read_csv(file_name, sep=';')
    except FileNotFoundError:
        st.error(f"Error: '{file_name}' not found. Please make sure it's in the same directory as this 'app.py' script.")
        return None
    except Exception as e:
        st.error(f"Error reading CSV: {e}. Check file format.")
        return None

    # 1. Clean Corrupted Data
    # Drop trailing empty columns/rows that are common in this dataset
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Drop NMHC(GT) which is mostly empty and not a primary target
    if 'NMHC(GT)' in df.columns:
        df.drop(columns=['NMHC(GT)'], inplace=True, errors='ignore')

    # 2. Create Datetime Index
    try:
        # Combine Date and Time columns
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
    except Exception as e:
        st.error(f"Error parsing datetime: {e}. Check Date/Time columns.")
        return None

    # 3. Handle Missing Values and Fix Data Types
    for col in df.columns:
        # Fix decimal format (e.g., '1,2' -> '1.2')
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        
        # Convert to numeric, replacing -200 with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(-200, np.nan)
        
    df.dropna(axis=1, how='all', inplace=True) # Drop any fully empty cols

    # 4. Impute Missing Data (Time-based)
    # Use time-based interpolation for missing sensor readings
    df.interpolate(method='time', inplace=True)
    # Use backfill to fill any NaNs at the very start of the dataset
    df.fillna(method='bfill', inplace=True)
    
    if df.isna().sum().sum() > 0:
        st.warning("Data still contains missing values after imputation. Dropping remaining NaN rows.")
        df.dropna(inplace=True)

    return df

# --- 2. Model Training & Analysis ---
# This function is cached to run only once
@st.cache_resource
def run_hybrid_analysis(_df):
    """
    Runs the full hybrid analysis:
    1. Trains IsolationForest for anomaly detection.
    2. Uses anomaly score as a feature to train RandomForestClassifier.
    3. Returns all models and results.
    """
    st.write("Cache: Training ML Models...")
    df = _df.copy()

    # --- Part 1: Anomaly Detection (IsolationForest) ---
    # We will check for anomalies in the main sensor readings
    met_features = {'T', 'RH', 'AH'}
    # Get all columns that are NOT meteorological
    anomaly_features = [col for col in df.columns if col not in met_features and col in df.columns]
    
    if not anomaly_features:
        st.error("No valid sensor features found for anomaly detection.")
        return None

    # Train the anomaly detection model
    iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(df[anomaly_features])
    
    # Create the anomaly_score as a new feature
    df['anomaly_score'] = iso_forest.decision_function(df[anomaly_features])
    
    # --- Part 2: Classification Forecast (RandomForest) ---
    
    # 1. Create Target Variable (y)
    # Define the thresholds based on your project idea
    def classify_aq(co):
        if co >= 4.0: return 2  # "Unhealthy"
        elif co >= 2.0: return 1  # "Moderate"
        else: return 0  # "Good"
    
    df['AQ_Class'] = df['CO(GT)'].apply(classify_aq)
    
    # This is the key forecasting step: predict the *next* hour's class
    df['target'] = df['AQ_Class'].shift(-1) 

    # 2. Create Feature Set (X)
    # Start with all features, including the new 'anomaly_score'
    features = list(df.columns.drop(['AQ_Class', 'target']))
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    features.extend(['hour', 'day_of_week'])
    
    # Add lag features (past values)
    df['CO_lag_1hr'] = df['CO(GT)'].shift(1)
    df['T_lag_1hr'] = df['T'].shift(1)
    features.extend(['CO_lag_1hr', 'T_lag_1hr'])

    # 3. Clean and Split
    # Drop rows with NaNs created by shift()
    df.dropna(inplace=True) 
    
    X = df[features]
    y = df['target'].astype(int)
    
    # Use an 80/20 split, respecting time order (no shuffle)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 4. Train Model
    rf_classifier = RandomForestClassifier(n_estimators=100, 
                                         random_state=42, 
                                         class_weight='balanced', # Helps with uneven class (Good/Moderate/Unhealthy)
                                         n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    
    # --- Part 3: Generate and Package Results ---
    y_pred = rf_classifier.predict(X_test)
    
    # Classification Report
    target_names = ['0: Good', '1: Moderate', '2: Unhealthy']
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                         index=[f"Actual {c}" for c in target_names], 
                         columns=[f"Pred {c}" for c in target_names])
    
    # Feature Importances
    importances = pd.Series(rf_classifier.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    
    # Anomaly Examples
    anomalies = df[df['anomaly_score'] < 0].sort_values(by='anomaly_score')
    
    results = {
        "report_df": report_df,
        "cm_df": cm_df,
        "importances": importances,
        "anomalies": anomalies,
        "total_anomalies": len(anomalies),
        "total_rows": len(_df)
    }
    return results

# --- 3. Streamlit App UI (The Report) ---

st.title("Hybrid Air Quality Analysis Report")
st.write("This report runs the complete analysis pipeline, including anomaly detection and next-hour classification, on the `AirQuality.csv` file.")

# Load data; if it fails, stop here.
df = load_and_clean_data('AirQuality.csv')

if df is not None:
    st.header("1. Historical Data")
    st.write(f"Loaded and cleaned {len(df)} data rows.")
    st.line_chart(df['CO(GT)'], use_container_width=True, color="#0068C9")
    with st.expander("Show Data Summary (df.describe())"):
        st.dataframe(df.describe())

    # Run the full analysis
    results = run_hybrid_analysis(df)

    if results:
        # --- Display Results ---
        st.header("2. Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Part 1: Anomaly Detection (IsolationForest)")
            st.metric(
                label="Anomalous Events Detected",
                value=f"{results['total_anomalies']}",
                help=f"Based on a 1% contamination setting. Total rows: {results['total_rows']}"
            )
            st.write("Most Anomalous Data Points (Lowest Scores):")
            st.dataframe(results['anomalies'][['CO(GT)', 'C6H6(GT)', 'T', 'anomaly_score']].head())

        with col2:
            st.subheader("Part 2: Next-Hour Risk Forecast (RandomForest)")
            st.write("Classification Report (on Test Set):")
            st.dataframe(results['report_df'])
            
            st.write("Confusion Matrix (Rows: Actual, Cols: Predicted):")
            st.dataframe(results['cm_df'])

        st.header("3. Key Drivers (Feature Importances)")
        st.write("These are the most important factors the classification model used to make its predictions.")
        
        # Highlight our hybrid feature
        try:
            # Find the rank of our custom feature
            anomaly_rank = list(results['importances'].index).index('anomaly_score') + 1
            st.info(f"**Key Insight:** The `anomaly_score` from Part 1 was the **#{anomaly_rank}** most important feature for predicting next-hour risk!")
        except ValueError:
            st.warning("'anomaly_score' was not found in the top features.")

        st.bar_chart(results['importances'].head(20))
        with st.expander("Show All Feature Importances (Raw Values)"):
            st.dataframe(results['importances'])

