# --------------------------------------------------
# Enhanced Air Quality Prediction using Random Forest
# --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1️⃣ Load and Clean Dataset
# --------------------------------------------------
df = pd.read_csv("AirQuality.csv", sep=';', low_memory=False)
df = df.dropna(how='all', axis=1)

# Create datetime index
df['Datetime'] = pd.to_datetime(df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip(),
                                format='%d/%m/%Y %H.%M.%S', errors='coerce')
df = df.set_index('Datetime').sort_index()

# Replace -200 with NaN (missing data marker)
df = df.replace(-200, np.nan)

# Clean numeric columns (remove stray spaces/commas)
def clean_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.replace(' ', '').str.replace(',', ''), errors='coerce')

for col in df.columns:
    if col in ['Date', 'Time']:
        continue
    df[col] = clean_numeric_series(df[col])

# Fill missing values with forward/backward fill
df = df.fillna(method='ffill').fillna(method='bfill')

print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# 2️⃣ Feature Selection
# --------------------------------------------------
features = ['T', 'RH', 'AH']       # meteorological factors
target = 'NO2(GT)'                 # pollutant to predict

# --------------------------------------------------
# 3️⃣ Feature Engineering (Lag & Rolling Mean)
# --------------------------------------------------
df_feat = df[features + [target]].copy()
for col in features + [target]:
    df_feat[f'{col}_lag1'] = df_feat[col].shift(1)
    df_feat[f'{col}_roll24'] = df_feat[col].rolling(window=24, min_periods=1).mean()

df_feat = df_feat.dropna()
engineered_features = [c for c in df_feat.columns if c != target]
X = df_feat[engineered_features]
y = df_feat[target]

# --------------------------------------------------
# 4️⃣ Train-Test Split (Time-based)
# --------------------------------------------------
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_index = df_feat.index[split_idx:]

# --------------------------------------------------
# 5️⃣ Scale Features
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 6️⃣ Train Random Forest Model
# --------------------------------------------------
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

# --------------------------------------------------
# 7️⃣ Evaluate Model
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Random Forest Evaluation Results:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# --------------------------------------------------
# 8️⃣ Visualization & Analysis
# --------------------------------------------------

# --- Graph 1: Correlation Heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(df_feat.corr()[[target]].sort_values(by=target, ascending=False), annot=True, cmap='coolwarm', fmt=".2f")
plt.title(f"Correlation of Meteorological Factors with {target}")
plt.tight_layout()
plt.show()
# 💡 What it shows: How strongly each meteorological parameter (T, RH, AH)
# correlates with NO2(GT). High positive means pollutant rises with that factor;
# negative means it decreases.

# --- Graph 2: Feature Importance ---
feat_importances = pd.Series(rf.feature_importances_, index=engineered_features).sort_values(ascending=True)
plt.figure(figsize=(9,6))
feat_importances.plot(kind='barh', color='teal')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
# 💡 What it shows: Which features (like T, RH, or AH lags/averages)
# had the biggest impact on NO2 predictions.

# --- Graph 3: Actual vs Predicted (Time Series) ---
plt.figure(figsize=(14,5))
plt.plot(test_index, y_test, label='Actual NO2(GT)', linewidth=1.8)
plt.plot(test_index, y_pred, label='Predicted NO2(GT)', linewidth=1.3)
plt.title("NO2(GT) - Actual vs Predicted Over Time")
plt.xlabel("Datetime")
plt.ylabel("NO2(GT) Concentration")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# 💡 What it shows: The model’s predictive accuracy over time.
# Closer alignment of both curves = better model performance.

# --- Graph 4: Scatter Plot (Predicted vs Actual) ---
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Predicted vs Actual NO2(GT)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.tight_layout()
plt.show()
# 💡 What it shows: Ideal model points lie along the red diagonal.
# The tighter the cluster around that line, the better the predictive performance.

# --- Graph 5: Temperature vs NO2 trend (Real-world insight) ---
plt.figure(figsize=(8,5))
sns.scatterplot(x=df_feat['T'], y=df_feat['NO2(GT)'], alpha=0.5)
sns.regplot(x=df_feat['T'], y=df_feat['NO2(GT)'], scatter=False, color='red')
plt.title("Relationship between Temperature and NO2(GT)")
plt.xlabel("Temperature (°C)")
plt.ylabel("NO2 Concentration")
plt.tight_layout()
plt.show()
# 💡 What it shows: Whether temperature increases or decreases correlate with pollutant levels.

# --------------------------------------------------
# 🔟 Comparison Table (Preview)
# --------------------------------------------------
compare_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=test_index)
print("\nFirst 10 Predictions vs Actual Values:")
print(compare_df.head(10))
