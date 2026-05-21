import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("loading parquet...")

df = pd.read_parquet("ml_data.parquet")

print(df.head())
print(df.shape)

df = df.sort_values(
    ["station_name", "date_parsed"]
)

df = df.head(150000)

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

        # TARGET FEATURES

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
        "no2_prev_6",

        # TIME FEATURES

        "hour",
        "day",
        "month",
        "dayofweek",
        "season",
        "is_night"
    ]

    # REMOVE DUPLICATES

    features = list(dict.fromkeys([
        f for f in features
        if f in df.columns
    ]))

    print("features:", features)

    # SPLIT DATA

    train_df = df[df["year"] < 2024]
    test_df = df[df["year"] == 2024]

    print("train shape:", train_df.shape)
    print("test shape:", test_df.shape)

    # SCALING

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_raw = scaler_x.fit_transform(
        train_df[features]
    )

    y_train_raw = scaler_y.fit_transform(
        train_df[[target]]
    )

    X_test_raw = scaler_x.transform(
        test_df[features]
    )

    y_test_raw = scaler_y.transform(
        test_df[[target]]
    )

    # SEQUENCES

    sequence_length = 24

    def make_sequences(X, y):

        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):

            X_seq.append(
                X[i-sequence_length:i]
            )

            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    print("creating train sequences...")

    X_train, y_train = make_sequences(
        X_train_raw,
        y_train_raw
    )

    print("creating test sequences...")

    X_test, y_test = make_sequences(
        X_test_raw,
        y_test_raw
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # TORCH TENSORS

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )

    # DATALOADER

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    # GRU MODEL

    class GRUModel(nn.Module):

        def __init__(self):

            super().__init__()

            self.gru = nn.GRU(
                input_size=len(features),
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                batch_first=True
            )

            self.fc = nn.Linear(64, 1)

        def forward(self, x):

            output, hidden = self.gru(x)

            hidden = hidden[-1]

            out = self.fc(hidden)

            return out

    model = GRUModel()

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # TRAINING

    print("training GRU...")

    epochs = 10

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_x, batch_y in train_loader:

            optimizer.zero_grad()

            pred = model(batch_x)

            loss = loss_fn(pred, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            "epoch:",
            epoch + 1,
            "loss:",
            total_loss
        )

    # SAVE MODEL

    print("saving GRU model...")

    torch.save(
        model.state_dict(),
        f"{target}_gru_model.pth"
    )

    print("GRU model saved")

    # PREDICTION

    print("predicting...")

    model.eval()

    with torch.no_grad():

        pred = model(X_test)

    pred = pred.numpy()
    real = y_test.numpy()

    pred = scaler_y.inverse_transform(pred)

    real = scaler_y.inverse_transform(real)

    # METRICS

    mae = mean_absolute_error(real, pred)

    mse = mean_squared_error(real, pred)

    rmse = mse ** 0.5

    r2 = r2_score(real, pred)

    print("GRU MAE:", mae)
    print("GRU MSE:", mse)
    print("GRU RMSE:", rmse)
    print("GRU R2:", r2)

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
        f"{target}_gru_results.csv",
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
        real[:1000],
        pred[:1000],
        alpha=0.5
    )

    plt.xlabel(f"Real {target.upper()}")
    plt.ylabel(f"Predicted {target.upper()}")

    plt.title(
        f"{target.upper()} GRU Forecast"
    )

    plt.tight_layout()

    plt.savefig(
        f"{target}_gru_scatter.png",
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
        f"{target.upper()} GRU R2 Score"
    )

    plt.ylabel("R2")

    plt.tight_layout()

    plt.savefig(
        f"{target}_gru_r2.png",
        dpi=300
    )

    plt.show()

# FINAL TABLE

final_results = pd.DataFrame(all_results)

print(final_results)

final_results.to_csv(
    "all_gru_results.csv",
    index=False
)

# GLOBAL COMPARISON

plt.figure(figsize=(10, 5))

plt.bar(
    final_results["Pollutant"],
    final_results["R2"]
)

plt.title(
    "GRU Comparison Across Pollutants"
)

plt.xlabel("Pollutant")
plt.ylabel("R2 Score")

plt.tight_layout()

plt.savefig(
    "gru_global_comparison.png",
    dpi=300
)

plt.show()

print("ALL GRU MODELS FINISHED")
