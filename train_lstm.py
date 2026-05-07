import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

print("loading parquet...")

df = pd.read_parquet("ml_data_sentinel.parquet")
df = df.head(150000)
df["landcover"] = df["landcover"].astype("category").cat.codes

features = [
    "pm25_prev_1",
    "pm25_rolling_3",
    "pm25_rolling_6",
    "hour",
    "day",
    "month",
    "landcover"
]

target = "pm25_future_1"

print("scaling...")

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(df[features])

y = scaler_y.fit_transform(
    df[[target]]
)

print("creating sequences...")

sequence_length = 24

X_seq = []
y_seq = []

for i in range(sequence_length, len(X)):
    X_seq.append(X[i-sequence_length:i])
    y_seq.append(y[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(X_seq.shape)
print(y_seq.shape)

split = int(len(X_seq) * 0.8)

X_train = X_seq[:split]
X_test = X_seq[split:]

y_train = y_seq[:split]
y_test = y_seq[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=len(features),
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        hidden = hidden[-1]

        out = self.fc(hidden)

        return out

model = LSTMModel()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("training LSTM...")

epochs = 5

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

    print("epoch:", epoch + 1, "loss:", total_loss)

print("predicting...")

model.eval()

with torch.no_grad():
    pred = model(X_test)

pred = pred.numpy()

pred = scaler_y.inverse_transform(pred)

real = scaler_y.inverse_transform(
    y_test.numpy()
)

mae = mean_absolute_error(real, pred)
r2 = r2_score(real, pred)

print("LSTM MAE:", mae)
print("LSTM R2:", r2)