import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from lightgbm import LGBMRegressor

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
        "pm25_prev_3",
        "pm25_prev_6",

        # PM10

        "pm10_prev_1",
        "pm10_prev_3",
        "pm10_prev_6",

        # NO2

        "no2_prev_1",
        "no2_prev_3",
        "no2_prev_6"
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

    # LIGHTGBM

    print("training LightGBM...")

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        f"{target}_lightgbm_model.pkl"
    )

    print("LightGBM trained")

    pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    mse = mean_squared_error(
        y_test,
        pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        pred
    )

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2:", r2)

    # RESULTS TABLE

    results = pd.DataFrame({
        "Metric": [
            "MAE",
            "MSE",
            "RMSE",
            "R2"
        ],
        "Value": [
            mae,
            mse,
            rmse,
            r2
        ]
    })

    print(results)

    results.to_csv(
        f"{target}_lightgbm_results.csv",
        index=False
    )

    # GLOBAL RESULTS

    all_results.append({
        "Pollutant": target.upper(),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    # FEATURE IMPORTANCE

    importance = model.feature_importances_

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
        f"{target.upper()} LightGBM Feature Importance"
    )

    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.xticks(rotation=60)

    plt.tight_layout()

    plt.savefig(
        f"{target}_lightgbm_feature_importance.png",
        dpi=300
    )

    plt.show()

    # SCATTER PLOT

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test[:1000],
        pred[:1000],
        alpha=0.5
    )

    plt.xlabel(f"Real {target.upper()}")
    plt.ylabel(f"Predicted {target.upper()}")

    plt.title(
        f"{target.upper()} LightGBM Forecast"
    )

    plt.tight_layout()

    plt.savefig(
        f"{target}_lightgbm_scatter.png",
        dpi=300
    )

    plt.show()

    # R2 GRAPH

    plt.figure(figsize=(8, 5))

    plt.bar(
        [target.upper()],
        [r2]
    )

    plt.title(
        f"{target.upper()} LightGBM R2 Score"
    )

    plt.ylabel("R2")

    plt.tight_layout()

    plt.savefig(
        f"{target}_lightgbm_r2.png",
        dpi=300
    )

    plt.show()

# FINAL COMPARISON TABLE

final_results = pd.DataFrame(all_results)

print(final_results)

final_results.to_csv(
    "all_lightgbm_results.csv",
    index=False
)

# GLOBAL R2 GRAPH

plt.figure(figsize=(10, 5))

plt.bar(
    final_results["Pollutant"],
    final_results["R2"]
)

plt.title("LightGBM Comparison Across Pollutants")

plt.xlabel("Pollutant")
plt.ylabel("R2 Score")

plt.tight_layout()

plt.savefig(
    "lightgbm_global_comparison.png",
    dpi=300
)

plt.show()

print("ALL LIGHTGBM MODELS FINISHED")
