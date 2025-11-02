import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # This is for saving our trained model

print("--- Starting Model Training ---")

# 1. Create a simple, dummy dataset for demonstration
# In a real project, you'd load a big CSV here.
data = {
    'Fever': [102, 98, 99, 101, 103, 98, 100, 99, 102, 98.5],
    'Cough': [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # 1 = Yes, 0 = No
    'Fatigue': [1, 0, 0, 1, 1, 0, 1, 0, 1, 0], # 1 = Yes, 0 = No
    'RunnyNose': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1], # 1 = Yes, 0 = No
    'Disease': ['Flu', 'Cold', 'Allergy', 'Cold', 'Flu', 'Allergy', 'Cold', 'Allergy', 'Flu', 'Cold']
}
df = pd.DataFrame(data)

print("Dummy dataset created:")
print(df.head())

# 2. Define Features (X) and Target (y)
X = df.drop('Disease', axis=1)
y = df['Disease']

# 3. Train the model
# (We skip train/test split for this simple demo, but you'd normally do it)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("\n--- Model has been trained ---")

# 4. Save the trained model to a file
# This is the most important part!
model_filename = 'disease_predictor_model.joblib'
joblib.dump(model, model_filename)

print(f"--- Model saved as {model_filename} ---")
print("You can now run the 'app.py' front-end.")