import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

        # TIME FEATURES

        "hour",
        "day",
        "month",
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

    # DNN

    print("training DNN...")

    dnn = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=100,
        random_state=42
    )

    dnn.fit(X_train, y_train)

    joblib.dump(
        dnn,
        f"{target}_dnn_markov_fuzzy_model.pkl"
    )

    print("DNN trained")
    print("DNN model saved")

    dnn_pred = dnn.predict(X_test)

    # MARKOV STATES

    print("building Markov states...")

    states = []

    for value in dnn_pred:

        if value < 35:
            states.append(0)

        elif value < 75:
            states.append(1)

        else:
            states.append(2)

    # FUZZY RULES

    print("applying fuzzy rules...")

    fuzzy_pred = []

    for pred, state in zip(dnn_pred, states):

        value = pred

        if state == 2:
            value = value * 1.1

        elif state == 0:
            value = value * 0.9

        fuzzy_pred.append(value)

    fuzzy_pred = np.array(fuzzy_pred)

    # METRICS

    mae = mean_absolute_error(
        y_test,
        fuzzy_pred
    )

    mse = mean_squared_error(
        y_test,
        fuzzy_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        fuzzy_pred
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
        f"{target}_dnn_markov_fuzzy_results.csv",
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

    # SCATTER PLOT

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test[:1000],
        fuzzy_pred[:1000],
        alpha=0.5
    )

    plt.xlabel(f"Real {target.upper()}")
    plt.ylabel(f"Predicted {target.upper()}")

    plt.title(
        f"{target.upper()} DNN + Markov + Fuzzy"
    )

    plt.tight_layout()

    plt.savefig(
        f"{target}_dnn_markov_fuzzy_scatter.png",
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
        f"{target.upper()} DNN + Markov + Fuzzy R2"
    )

    plt.ylabel("R2")

    plt.tight_layout()

    plt.savefig(
        f"{target}_dnn_markov_fuzzy_r2.png",
        dpi=300
    )

    plt.show()

# FINAL TABLE

final_results = pd.DataFrame(all_results)

print(final_results)

final_results.to_csv(
    "all_dnn_markov_fuzzy_results.csv",
    index=False
)

# GLOBAL COMPARISON

plt.figure(figsize=(10, 5))

plt.bar(
    final_results["Pollutant"],
    final_results["R2"]
)

plt.title(
    "DNN + Markov + Fuzzy Comparison"
)

plt.xlabel("Pollutant")
plt.ylabel("R2 Score")

plt.tight_layout()

plt.savefig(
    "dnn_markov_fuzzy_global_comparison.png",
    dpi=300
)

plt.show()

print("ALL DNN MODELS FINISHED")
