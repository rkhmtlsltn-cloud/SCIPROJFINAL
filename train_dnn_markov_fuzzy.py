import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

print("loading parquet...")

df = pd.read_parquet("ml_data.parquet")

features = [
    "pm25_prev_1",
    "pm25_rolling_3",
    "pm25_rolling_6",
    "hour",
    "day",
    "month"
]

target = "pm25_future_1"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("training DNN...")

dnn = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=20,
    random_state=42
)

dnn.fit(X_train, y_train)

print("DNN trained")

dnn_pred = dnn.predict(X_test)

print("building Markov states...")

states = []

for value in dnn_pred:

    if value < 35:
        states.append(0)

    elif value < 75:
        states.append(1)

    else:
        states.append(2)

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

mae = mean_absolute_error(y_test, fuzzy_pred)
r2 = r2_score(y_test, fuzzy_pred)

print("DNN + Markov + Fuzzy MAE:", mae)
print("DNN + Markov + Fuzzy R2:", r2)