import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

df = pd.read_csv("wait_dataset.csv")
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]-1} features")
print(f"Features: {[c for c in df.columns if c != 'wait_time']}\n")

X = df.drop("wait_time", axis=1)
y = df["wait_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MAE:      {mean_absolute_error(y_test, y_pred):.2f} timesteps")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

with open("wait_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved as wait_model.pkl")

print("\nFeature importances:")
feature_names = list(X.columns)
importance = model.feature_importances_
for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:<22} {imp:.4f}  {bar}")
