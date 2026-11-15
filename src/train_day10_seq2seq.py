from pathlib import Path
import math

import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from day10_attention import TemporalAttention


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42

STORE_ID = 1
ITEM_ID = 1

LOOK_BACK = 30

FORECAST_HORIZON = 7

TEST_DAYS = 35

BATCH_SIZE = 16
MAX_EPOCHS = 40

DATA_PATH = Path(
    "data/day9/train.csv"
)

MODEL_DIRECTORY = Path(
    "models"
)

OUTPUT_DIRECTORY = Path(
    "outputs"
)

MODEL_PATH = (
    MODEL_DIRECTORY /
    "day10_seq2seq.keras"
)

RESULTS_PATH = (
    MODEL_DIRECTORY /
    "day10_results.csv"
)

SCALING_PATH = (
    MODEL_DIRECTORY /
    "day10_scaling.csv"
)

LAST_WINDOW_PATH = (
    MODEL_DIRECTORY /
    "day10_last_window.csv"
)

MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

tf.keras.utils.set_random_seed(
    RANDOM_STATE
)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

if not DATA_PATH.exists():

    raise FileNotFoundError(
        "data/day9/train.csv was not found."
    )

df = pd.read_csv(
    DATA_PATH
)

required_columns = [
    "date",
    "store",
    "item",
    "sales"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


# --------------------------------------------------
# 3. Prepare data types
# --------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["sales"] = pd.to_numeric(
    df["sales"],
    errors="coerce"
)

df = df.dropna(
    subset=required_columns
)


# --------------------------------------------------
# 4. Select same series as Day 9
# --------------------------------------------------

series_df = df[
    (df["store"] == STORE_ID) &
    (df["item"] == ITEM_ID)
][
    ["date", "sales"]
].copy()

if series_df.empty:

    raise ValueError(
        "Selected store/item has no data."
    )


# --------------------------------------------------
# 5. Daily time series
# --------------------------------------------------

series_df = (
    series_df
    .groupby(
        "date",
        as_index=False
    )["sales"]
    .sum()
    .sort_values("date")
)

series_df = (
    series_df
    .set_index("date")
    .resample("D")
    .sum()
    .reset_index()
)

series_df["sales"] = (
    series_df["sales"]
    .fillna(0)
    .astype("float32")
)

print("=" * 60)

print(
    "DAY 10: SEQ2SEQ + ATTENTION"
)

print("=" * 60)

print(
    "Time-series rows:",
    len(series_df)
)


# --------------------------------------------------
# 6. Time-based split
# --------------------------------------------------

split_index = (
    len(series_df) -
    TEST_DAYS
)

training_sales = (
    series_df["sales"]
    .iloc[:split_index]
)

sales_minimum = float(
    training_sales.min()
)

sales_maximum = float(
    training_sales.max()
)

sales_range = (
    sales_maximum -
    sales_minimum
)

if sales_range == 0:

    raise ValueError(
        "Sales have no variation."
    )


# --------------------------------------------------
# 7. Scale
# --------------------------------------------------

series_df["scaled_sales"] = (
    (
        series_df["sales"] -
        sales_minimum
    ) /
    sales_range
)


# --------------------------------------------------
# 8. Create multi-step windows
# --------------------------------------------------

def create_windows(
    values,
    look_back,
    horizon
):

    X = []
    y = []
    indexes = []

    final_index = (
        len(values) -
        horizon +
        1
    )

    for index in range(
        look_back,
        final_index
    ):

        history = values[
            index - look_back:
            index
        ]

        future = values[
            index:
            index + horizon
        ]

        X.append(
            history
        )

        y.append(
            future
        )

        indexes.append(
            index
        )

    return X, y, indexes


values = (
    series_df["scaled_sales"]
    .tolist()
)

X_all, y_all, indexes = (
    create_windows(
        values,
        LOOK_BACK,
        FORECAST_HORIZON
    )
)


# --------------------------------------------------
# 9. Separate train/test
# --------------------------------------------------

X_train_list = []
y_train_list = []

X_test_list = []
y_test_list = []

for X_window, y_window, index in zip(
    X_all,
    y_all,
    indexes
):

    # All forecast dates must stay
    # inside the training period.
    if (
        index +
        FORECAST_HORIZON
        <= split_index
    ):

        X_train_list.append(
            X_window
        )

        y_train_list.append(
            y_window
        )

    elif index >= split_index:

        X_test_list.append(
            X_window
        )

        y_test_list.append(
            y_window
        )


# --------------------------------------------------
# 10. Tensors
# --------------------------------------------------

X_train = tf.convert_to_tensor(
    X_train_list,
    dtype=tf.float32
)

y_train = tf.convert_to_tensor(
    y_train_list,
    dtype=tf.float32
)

X_test = tf.convert_to_tensor(
    X_test_list,
    dtype=tf.float32
)

y_test = tf.convert_to_tensor(
    y_test_list,
    dtype=tf.float32
)

X_train = tf.expand_dims(
    X_train,
    axis=-1
)

X_test = tf.expand_dims(
    X_test,
    axis=-1
)

print(
    "Training windows:",
    X_train.shape
)

print(
    "Testing windows:",
    X_test.shape
)


# --------------------------------------------------
# 11. Build encoder
# --------------------------------------------------

inputs = Input(
    shape=(
        LOOK_BACK,
        1
    ),
    name="sales_history"
)

encoder_sequence = LSTM(
    32,
    return_sequences=True,
    name="encoder_lstm"
)(inputs)


# --------------------------------------------------
# 12. Attention
# --------------------------------------------------

context, attention_weights = (
    TemporalAttention(
        name="temporal_attention"
    )(
        encoder_sequence
    )
)


# --------------------------------------------------
# 13. Forecast head / decoder
# --------------------------------------------------

x = Dense(
    32,
    activation="relu"
)(context)

x = Dropout(
    0.20
)(x)

forecast = Dense(
    FORECAST_HORIZON,
    name="multi_step_forecast"
)(x)


# --------------------------------------------------
# 14. Training model
# --------------------------------------------------

model = Model(
    inputs=inputs,
    outputs=forecast,
    name="day10_attention_forecaster"
)


# --------------------------------------------------
# 15. Attention inspection model
# --------------------------------------------------

attention_model = Model(
    inputs=inputs,
    outputs=attention_weights
)


# --------------------------------------------------
# 16. Compile
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# --------------------------------------------------
# 17. Early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# --------------------------------------------------
# 18. Train
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=[
        early_stopping
    ],
    verbose=1
)


# --------------------------------------------------
# 19. Test predictions
# --------------------------------------------------

scaled_predictions = model.predict(
    X_test,
    verbose=0
)

scaled_actual = (
    y_test.numpy()
)


# --------------------------------------------------
# 20. Convert to real sales
# --------------------------------------------------

predictions = (
    scaled_predictions *
    sales_range +
    sales_minimum
)

actual = (
    scaled_actual *
    sales_range +
    sales_minimum
)

predictions = tf.maximum(
    predictions,
    0
).numpy()


# --------------------------------------------------
# 21. Evaluate all horizons
# --------------------------------------------------

absolute_errors = abs(
    actual -
    predictions
)

squared_errors = (
    actual -
    predictions
) ** 2

mae = float(
    absolute_errors.mean()
)

rmse = math.sqrt(
    float(
        squared_errors.mean()
    )
)


# --------------------------------------------------
# 22. Per-horizon metrics
# --------------------------------------------------

result_rows = []

for horizon in range(
    FORECAST_HORIZON
):

    horizon_mae = float(
        absolute_errors[
            :,
            horizon
        ].mean()
    )

    horizon_rmse = math.sqrt(
        float(
            squared_errors[
                :,
                horizon
            ].mean()
        )
    )

    result_rows.append({
        "Horizon":
            f"Day +{horizon + 1}",

        "MAE":
            round(
                horizon_mae,
                4
            ),

        "RMSE":
            round(
                horizon_rmse,
                4
            )
    })


results_df = pd.DataFrame(
    result_rows
)


# --------------------------------------------------
# 23. Overall row
# --------------------------------------------------

overall_row = pd.DataFrame([
    {
        "Horizon": "Overall",
        "MAE": round(
            mae,
            4
        ),
        "RMSE": round(
            rmse,
            4
        )
    }
])

results_df = pd.concat(
    [
        results_df,
        overall_row
    ],
    ignore_index=True
)


# --------------------------------------------------
# 24. Print
# --------------------------------------------------

print("\n" + "=" * 60)

print(
    "DAY 10 MULTI-HORIZON RESULTS"
)

print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 25. Save model
# --------------------------------------------------

model.save(
    MODEL_PATH
)


# --------------------------------------------------
# 26. Save scaling details
# --------------------------------------------------

pd.DataFrame([
    {
        "Store": STORE_ID,
        "Item": ITEM_ID,
        "Look Back": LOOK_BACK,
        "Forecast Horizon":
            FORECAST_HORIZON,
        "Minimum Sales":
            sales_minimum,
        "Maximum Sales":
            sales_maximum
    }
]).to_csv(
    SCALING_PATH,
    index=False
)


# --------------------------------------------------
# 27. Save final history window
# --------------------------------------------------

last_window = series_df.tail(
    LOOK_BACK
)[
    [
        "date",
        "sales",
        "scaled_sales"
    ]
]

last_window.to_csv(
    LAST_WINDOW_PATH,
    index=False
)


# --------------------------------------------------
# 28. Save metrics
# --------------------------------------------------

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# --------------------------------------------------
# 29. Get attention for latest test sample
# --------------------------------------------------

sample = X_test[-1:]

weights = attention_model.predict(
    sample,
    verbose=0
)[0]

weights = tf.squeeze(
    weights
).numpy()


# --------------------------------------------------
# 30. Save attention information
# --------------------------------------------------

attention_df = pd.DataFrame({
    "History Day": [
        f"Day -{LOOK_BACK - index}"
        for index in range(
            LOOK_BACK
        )
    ],

    "Attention Weight":
        weights
})

attention_df.to_csv(
    MODEL_DIRECTORY /
    "day10_attention_weights.csv",
    index=False
)


# --------------------------------------------------
# 31. Final
# --------------------------------------------------

print("\nGenerated files:")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Results:",
    RESULTS_PATH
)

print(
    "Attention weights:",
    MODEL_DIRECTORY /
    "day10_attention_weights.csv"
)