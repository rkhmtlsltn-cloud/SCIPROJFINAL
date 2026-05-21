import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

print("loading parquet...")

df = pd.read_parquet("ml_data.parquet")

print(df.head())
print(df.shape)

pollutants = [
    "pm25",
    "pm10",
    "no2"
]

all_results = []

for target in pollutants:

    print("\n========================")
    print("TARGET:", target)
    print("========================")

    features = [

        "latitude",
        "longitude",

        "year",
        "month",
        "day",
        "hour",
        "dayofweek",

        "season",
        "is_night",

        # TARGET FEATURE

        f"{target}_prev_1",

        # PM25

        "pm25_prev_1",
        "pm25_rolling_3",

        # PM10

        "pm10_prev_1",
        "pm10_rolling_3",

        # NO2

        "no2_prev_1",
        "no2_rolling_3"
    ]

    # REMOVE DUPLICATES

    features = list(dict.fromkeys([
        f for f in features
        if f in df.columns
    ]))

    print("features:", features)

    print("splitting by years...")

    train_df = df[df["year"] < 2024]
    test_df = df[df["year"] == 2024]

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    print("train shape:", train_df.shape)
    print("test shape:", test_df.shape)

    # RANDOM FOREST

    print("training RF...")

    rf = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    joblib.dump(
        rf,
        f"{target}_random_forest_model.pkl"
    )

    print("RF trained")

    rf_pred = rf.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_rmse = rf_mse ** 0.5
    rf_r2 = r2_score(y_test, rf_pred)

    print("RF MAE:", rf_mae)
    print("RF MSE:", rf_mse)
    print("RF RMSE:", rf_rmse)
    print("RF R2:", rf_r2)

    # XGBOOST

    print("training XGBoost...")

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

    joblib.dump(
        xgb,
        f"{target}_xgboost_model.pkl"
    )

    print("XGBoost trained")

    xgb_pred = xgb.predict(X_test)

    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_rmse = xgb_mse ** 0.5
    xgb_r2 = r2_score(y_test, xgb_pred)

    print("XGB MAE:", xgb_mae)
    print("XGB MSE:", xgb_mse)
    print("XGB RMSE:", xgb_rmse)
    print("XGB R2:", xgb_r2)

    # HYBRIDs

    print("making hybrid...")

    hybrid_pred = (
        rf_pred * 0.8
    ) + (
        xgb_pred * 0.2
    )

    hybrid_mae = mean_absolute_error(
        y_test,
        hybrid_pred
    )

    hybrid_mse = mean_squared_error(
        y_test,
        hybrid_pred
    )

    hybrid_rmse = hybrid_mse ** 0.5

    hybrid_r2 = r2_score(
        y_test,
        hybrid_pred
    )

    print("Hybrid MAE:", hybrid_mae)
    print("Hybrid MSE:", hybrid_mse)
    print("Hybrid RMSE:", hybrid_rmse)
    print("Hybrid R2:", hybrid_r2)

    # RESULTS TABLE

    results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost",
            "Hybrid"
        ],
        "MAE": [
            rf_mae,
            xgb_mae,
            hybrid_mae
        ],
        "MSE": [
            rf_mse,
            xgb_mse,
            hybrid_mse
        ],
        "RMSE": [
            rf_rmse,
            xgb_rmse,
            hybrid_rmse
        ],
        "R2": [
            rf_r2,
            xgb_r2,
            hybrid_r2
        ]
    })

    print(results)

    results.to_csv(
        f"{target}_results.csv",
        index=False
    )

    # SAVE GLOBAL RESULTS

    for i in range(len(results)):

        all_results.append({
            "Pollutant": target.upper(),
            "Model": results.iloc[i]["Model"],
            "MAE": results.iloc[i]["MAE"],
            "MSE": results.iloc[i]["MSE"],
            "RMSE": results.iloc[i]["RMSE"],
            "R2": results.iloc[i]["R2"]
        })

    # FEATURE IMPORTANCE

    importance = rf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance_df)

    plt.figure(figsize=(12, 5))

    plt.bar(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.title(
        f"{target.upper()} Feature Importance"
    )

    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.xticks(rotation=60)

    plt.tight_layout()

    plt.savefig(
        f"{target}_feature_importance.png",
        dpi=300
    )

    plt.show()

    # CORRELATION MATRIX

    corr = df[features + [target]].corr()

    plt.figure(figsize=(14, 12))

    sns.heatmap(
        corr,
        cmap="coolwarm"
    )

    plt.title(
        f"{target.upper()} Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        f"{target}_correlation_matrix.png",
        dpi=300
    )

    plt.show()

    # RF SCATTER

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test[:1000],
        rf_pred[:1000],
        alpha=0.5
    )

    plt.xlabel(f"Real {target.upper()}")
    plt.ylabel(f"Predicted {target.upper()}")

    plt.title(
        f"{target.upper()} Random Forest"
    )

    plt.tight_layout()

    plt.savefig(
        f"{target}_rf_scatter.png",
        dpi=300
    )

    plt.show()

    # XGB SCATTER

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test[:1000],
        xgb_pred[:1000],
        alpha=0.5
    )

    plt.xlabel(f"Real {target.upper()}")
    plt.ylabel(f"Predicted {target.upper()}")

    plt.title(
        f"{target.upper()} XGBoost"
    )

    plt.tight_layout()

    plt.savefig(
        f"{target}_xgb_scatter.png",
        dpi=300
    )

    plt.show()

    # HYBRID SCATTER

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test[:1000],
        hybrid_pred[:1000],
        alpha=0.5
    )

    plt.xlabel(f"Real {target.upper()}")
    plt.ylabel(f"Predicted {target.upper()}")

    plt.title(
        f"{target.upper()} Hybrid Forecast"
    )

    plt.tight_layout()

    plt.savefig(
        f"{target}_hybrid_scatter.png",
        dpi=300
    )

    plt.show()

    # R2 COMPARISON

    plt.figure(figsize=(8, 5))

    plt.bar(
        results["Model"],
        results["R2"]
    )

    plt.title(
        f"{target.upper()} Model Comparison"
    )

    plt.xlabel("Model")
    plt.ylabel("R2 Score")

    plt.tight_layout()

    plt.savefig(
        f"{target}_r2_comparison.png",
        dpi=300
    )

    plt.show()

# FINAL GLOBAL RESULTS

final_results = pd.DataFrame(all_results)

print(final_results)

final_results.to_csv(
    "all_pollutants_comparison.csv",
    index=False
)

# GLOBAL COMPARISON GRAPH

plt.figure(figsize=(12, 6))

for pollutant in final_results["Pollutant"].unique():

    subset = final_results[
        final_results["Pollutant"] == pollutant
    ]

    plt.plot(
        subset["Model"],
        subset["R2"],
        marker="o",
        label=pollutant
    )

plt.title("Model Comparison Across Pollutants")

plt.xlabel("Models")
plt.ylabel("R2 Score")

plt.legend()

plt.tight_layout()

plt.savefig(
    "global_model_comparison.png",
    dpi=300
)

plt.show()

print("ALL MODELS FINISHED")
