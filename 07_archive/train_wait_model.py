import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

df = pd.read_csv("wait_dataset.csv")

X = df.drop("wait_time", axis=1)
y = df["wait_time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# save model
with open("wait_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as wait_model.pkl")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

feature_names = ["job_gpu", "total_free", "queue_length", "running_jobs",
                 "max_free_node", "variance_free", "smaller_jobs_in_queue", "demand_ratio"]

importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

print("\nFeature Importance:")
for i in sorted_idx:
    print(f"  {feature_names[i]}: {importance[i]:.4f}")

plt.figure(figsize=(8, 5))
plt.bar([feature_names[i] for i in sorted_idx], importance[sorted_idx])
plt.xticks(rotation=45, ha='right')
plt.title("Feature Importance for Wait Time Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\nPlot saved as feature_importance.png")