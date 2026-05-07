import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

print("loading sentinel parquet...")

df = pd.read_parquet("ml_data_sentinel.parquet")

df["landcover"] = df["landcover"].astype("category").cat.codes

print(df.head())
print(df.shape)

features = [
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "hour",
    "dayofweek",
    "pm25_prev_1",
    "pm25_rolling_3",
    "pm25_rolling_6",
    "landcover"
]

target = "pm25_future_1"

X = df[features]
y = df[target]

print("splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("training RF...")

rf = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("RF trained")

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("RF MAE:", rf_mae)
print("RF R2:", rf_r2)

print("training Sentinel + XGBoost...")

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

print("Sentinel + XGBoost trained")

xgb_pred = xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print("Sentinel + XGBoost MAE:", xgb_mae)
print("Sentinel + XGBoost R2:", xgb_r2)

print("making Sentinel hybrid...")

hybrid_pred = (rf_pred * 0.8) + (xgb_pred * 0.2)

hybrid_mae = mean_absolute_error(y_test, hybrid_pred)
hybrid_r2 = r2_score(y_test, hybrid_pred)

print("Sentinel Hybrid MAE:", hybrid_mae)
print("Sentinel Hybrid R2:", hybrid_r2)

results = pd.DataFrame({
    "Model": [
        "Random Forest + Sentinel",
        "XGBoost + Sentinel",
        "RF + XGBoost + Sentinel"
    ],
    "MAE": [
        rf_mae,
        xgb_mae,
        hybrid_mae
    ],
    "R2": [
        rf_r2,
        xgb_r2,
        hybrid_r2
    ]
})

print(results)

results.to_csv("sentinel_model_results.csv", index=False)

importance = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance_df)

plt.figure(figsize=(9, 5))
plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.title("Sentinel Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("sentinel_feature_importance.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    y_test[:1000],
    xgb_pred[:1000],
    alpha=0.5
)

plt.xlabel("Real PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Sentinel + XGBoost: Real vs Predicted PM2.5")
plt.tight_layout()
plt.savefig("sentinel_xgboost_real_vs_predicted.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["R2"])
plt.title("Sentinel Model R2 Comparison")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("sentinel_r2_comparison.png", dpi=300)
plt.show()