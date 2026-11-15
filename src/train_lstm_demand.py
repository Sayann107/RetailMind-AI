from pathlib import Path
import math

import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42

STORE_ID = 1
ITEM_ID = 1

LOOK_BACK = 30
TEST_DAYS = 30

BATCH_SIZE = 16
MAX_EPOCHS = 50

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
    "day9_lstm.keras"
)

SCALING_PATH = (
    MODEL_DIRECTORY /
    "day9_scaling.csv"
)

RESULTS_PATH = (
    MODEL_DIRECTORY /
    "day9_results.csv"
)

PREDICTIONS_PATH = (
    MODEL_DIRECTORY /
    "day9_test_predictions.csv"
)

LAST_WINDOW_PATH = (
    MODEL_DIRECTORY /
    "day9_last_window.csv"
)

PLOT_PATH = (
    OUTPUT_DIRECTORY /
    "day9_actual_vs_predicted.svg"
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
# 2. Check dataset
# --------------------------------------------------

if not DATA_PATH.exists():

    raise FileNotFoundError(
        "\nDay 9 dataset was not found:\n"
        "data/day9/train.csv\n\n"
        "Place the Store Item Demand "
        "train.csv file in data/day9/."
    )


# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH
)

print("=" * 60)
print("DAY 9: LSTM DEMAND FORECASTING")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 4. Validate columns
# --------------------------------------------------

required_columns = [
    "date",
    "store",
    "item",
    "sales"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# --------------------------------------------------
# 5. Convert date column
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
    subset=[
        "date",
        "store",
        "item",
        "sales"
    ]
)


# --------------------------------------------------
# 6. Select one store and item
# --------------------------------------------------

series_df = df[
    (df["store"] == STORE_ID) &
    (df["item"] == ITEM_ID)
].copy()

if series_df.empty:

    raise ValueError(
        f"No records found for "
        f"store={STORE_ID}, item={ITEM_ID}."
    )

print(
    f"\nSelected store: {STORE_ID}"
)

print(
    f"Selected item : {ITEM_ID}"
)

print(
    "Selected records:",
    len(series_df)
)


# --------------------------------------------------
# 7. Create a clean daily series
# --------------------------------------------------

series_df = (
    series_df[
        ["date", "sales"]
    ]
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

# Fill missing sales dates with zero
series_df["sales"] = (
    series_df["sales"]
    .fillna(0.0)
    .astype("float32")
)

print(
    "\nTime-series rows:",
    len(series_df)
)

print(
    "First date:",
    series_df["date"].min()
)

print(
    "Last date:",
    series_df["date"].max()
)


# --------------------------------------------------
# 8. Check series length
# --------------------------------------------------

minimum_rows = (
    LOOK_BACK +
    TEST_DAYS +
    10
)

if len(series_df) < minimum_rows:

    raise ValueError(
        f"At least {minimum_rows} daily "
        "records are required."
    )


# --------------------------------------------------
# 9. Train-test split by time
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
        "The selected sales series has "
        "no variation."
    )


# --------------------------------------------------
# 10. Scale sales manually
# --------------------------------------------------

series_df["scaled_sales"] = (
    (
        series_df["sales"] -
        sales_minimum
    ) /
    sales_range
)


# --------------------------------------------------
# 11. Window creation function
# --------------------------------------------------

def create_windows(values, look_back):

    inputs = []
    targets = []
    target_indexes = []

    for index in range(
        look_back,
        len(values)
    ):

        inputs.append(
            values[
                index - look_back:
                index
            ]
        )

        targets.append(
            values[index]
        )

        target_indexes.append(
            index
        )

    return (
        inputs,
        targets,
        target_indexes
    )


# --------------------------------------------------
# 12. Create supervised windows
# --------------------------------------------------

scaled_values = (
    series_df["scaled_sales"]
    .tolist()
)

all_inputs, all_targets, indexes = (
    create_windows(
        scaled_values,
        LOOK_BACK
    )
)


# --------------------------------------------------
# 13. Separate train and test windows
# --------------------------------------------------

X_train_list = []
y_train_list = []

X_test_list = []
y_test_list = []
test_indexes = []

for features, target, target_index in zip(
    all_inputs,
    all_targets,
    indexes
):

    if target_index < split_index:

        X_train_list.append(
            features
        )

        y_train_list.append(
            target
        )

    else:

        X_test_list.append(
            features
        )

        y_test_list.append(
            target
        )

        test_indexes.append(
            target_index
        )


# --------------------------------------------------
# 14. Convert to tensors
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


# --------------------------------------------------
# 15. Add feature dimension
# --------------------------------------------------

X_train = tf.expand_dims(
    X_train,
    axis=-1
)

X_test = tf.expand_dims(
    X_test,
    axis=-1
)

print(
    "\nX_train shape:",
    X_train.shape
)

print(
    "X_test shape :",
    X_test.shape
)


# --------------------------------------------------
# 16. Build LSTM
# --------------------------------------------------

model = Sequential([
    Input(
        shape=(
            LOOK_BACK,
            1
        )
    ),

    LSTM(
        32,
        return_sequences=True
    ),

    Dropout(0.20),

    LSTM(
        16
    ),

    Dropout(0.20),

    Dense(
        8,
        activation="relu"
    ),

    Dense(
        1
    )
])


# --------------------------------------------------
# 17. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("\nLSTM architecture:")

model.summary()


# --------------------------------------------------
# 18. Early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# --------------------------------------------------
# 19. Train LSTM
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)


# --------------------------------------------------
# 20. Predict test period
# --------------------------------------------------

scaled_predictions = (
    model.predict(
        X_test,
        verbose=0
    )
    .reshape(-1)
)

scaled_actual = (
    y_test.numpy()
    .reshape(-1)
)


# --------------------------------------------------
# 21. Convert back to sales values
# --------------------------------------------------

predicted_sales = (
    scaled_predictions *
    sales_range +
    sales_minimum
)

actual_sales = (
    scaled_actual *
    sales_range +
    sales_minimum
)

# Demand cannot be negative
predicted_sales = [
    max(0.0, float(value))
    for value in predicted_sales
]

actual_sales = [
    float(value)
    for value in actual_sales
]


# --------------------------------------------------
# 22. Calculate MAE manually
# --------------------------------------------------

absolute_errors = [
    abs(actual - predicted)
    for actual, predicted
    in zip(
        actual_sales,
        predicted_sales
    )
]

mae = sum(
    absolute_errors
) / len(
    absolute_errors
)


# --------------------------------------------------
# 23. Calculate RMSE manually
# --------------------------------------------------

squared_errors = [
    (actual - predicted) ** 2
    for actual, predicted
    in zip(
        actual_sales,
        predicted_sales
    )
]

mse = sum(
    squared_errors
) / len(
    squared_errors
)

rmse = math.sqrt(
    mse
)


# --------------------------------------------------
# 24. Create prediction result
# --------------------------------------------------

test_dates = [
    series_df["date"].iloc[index]
    for index in test_indexes
]

prediction_df = pd.DataFrame({
    "Date": test_dates,
    "Actual Sales": actual_sales,
    "Predicted Sales": predicted_sales,
    "Absolute Error": absolute_errors
})

prediction_df["Actual Sales"] = (
    prediction_df["Actual Sales"]
    .round(2)
)

prediction_df["Predicted Sales"] = (
    prediction_df["Predicted Sales"]
    .round(2)
)

prediction_df["Absolute Error"] = (
    prediction_df["Absolute Error"]
    .round(2)
)


# --------------------------------------------------
# 25. Print results
# --------------------------------------------------

print("\n" + "=" * 60)
print("DAY 9 LSTM TEST RESULTS")
print("=" * 60)

print(
    f"Store ID : {STORE_ID}"
)

print(
    f"Item ID  : {ITEM_ID}"
)

print(
    f"Look-back: {LOOK_BACK} days"
)

print(
    f"Test days: {len(prediction_df)}"
)

print(
    f"MAE      : {mae:.4f}"
)

print(
    f"RMSE     : {rmse:.4f}"
)

print("\nPrediction sample:")

print(
    prediction_df.head().to_string(
        index=False
    )
)


# --------------------------------------------------
# 26. Save trained model
# --------------------------------------------------

model.save(
    MODEL_PATH
)


# --------------------------------------------------
# 27. Save scaling information
# --------------------------------------------------

scaling_df = pd.DataFrame([
    {
        "Store": STORE_ID,
        "Item": ITEM_ID,
        "Look Back": LOOK_BACK,
        "Minimum Sales": sales_minimum,
        "Maximum Sales": sales_maximum
    }
])

scaling_df.to_csv(
    SCALING_PATH,
    index=False
)


# --------------------------------------------------
# 28. Save result summary
# --------------------------------------------------

results_df = pd.DataFrame([
    {
        "Model": "LSTM",
        "Store": STORE_ID,
        "Item": ITEM_ID,
        "Look Back Days": LOOK_BACK,
        "Test Days": len(
            prediction_df
        ),
        "MAE": round(
            mae,
            4
        ),
        "RMSE": round(
            rmse,
            4
        ),
        "Epochs Completed": len(
            history.history["loss"]
        )
    }
])

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# --------------------------------------------------
# 29. Save test predictions
# --------------------------------------------------

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# --------------------------------------------------
# 30. Save final look-back window
# --------------------------------------------------

last_window_df = series_df[
    ["date", "sales", "scaled_sales"]
].tail(
    LOOK_BACK
)

last_window_df.to_csv(
    LAST_WINDOW_PATH,
    index=False
)


# --------------------------------------------------
# 31. Create lightweight SVG plot
# --------------------------------------------------

def create_svg_plot(
    dates,
    actual_values,
    predicted_values,
    output_path
):

    width = 1000
    height = 500

    left = 70
    right = 30
    top = 50
    bottom = 70

    plot_width = (
        width - left - right
    )

    plot_height = (
        height - top - bottom
    )

    all_values = (
        list(actual_values) +
        list(predicted_values)
    )

    minimum = min(all_values)
    maximum = max(all_values)

    if maximum == minimum:
        maximum = minimum + 1

    def x_position(index):

        if len(dates) == 1:
            return left

        return (
            left +
            index *
            plot_width /
            (len(dates) - 1)
        )

    def y_position(value):

        return (
            top +
            (
                maximum - value
            ) *
            plot_height /
            (
                maximum - minimum
            )
        )

    actual_points = " ".join(
        f"{x_position(index):.2f},"
        f"{y_position(value):.2f}"
        for index, value
        in enumerate(actual_values)
    )

    predicted_points = " ".join(
        f"{x_position(index):.2f},"
        f"{y_position(value):.2f}"
        for index, value
        in enumerate(predicted_values)
    )

    svg = f"""<svg
xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">

<rect
width="100%"
height="100%"
fill="white"/>

<text
x="{width / 2}"
y="28"
text-anchor="middle"
font-family="Arial"
font-size="20">
Day 9: Actual vs Predicted Demand
</text>

<line
x1="{left}"
y1="{top}"
x2="{left}"
y2="{height - bottom}"
stroke="black"/>

<line
x1="{left}"
y1="{height - bottom}"
x2="{width - right}"
y2="{height - bottom}"
stroke="black"/>

<polyline
points="{actual_points}"
fill="none"
stroke="#1565C0"
stroke-width="3"/>

<polyline
points="{predicted_points}"
fill="none"
stroke="#E53935"
stroke-width="3"/>

<line
x1="700"
y1="25"
x2="740"
y2="25"
stroke="#1565C0"
stroke-width="3"/>

<text
x="750"
y="30"
font-family="Arial"
font-size="14">
Actual
</text>

<line
x1="830"
y1="25"
x2="870"
y2="25"
stroke="#E53935"
stroke-width="3"/>

<text
x="880"
y="30"
font-family="Arial"
font-size="14">
Predicted
</text>

<text
x="20"
y="{height / 2}"
font-family="Arial"
font-size="14"
transform="rotate(-90 20 {height / 2})"
text-anchor="middle">
Sales
</text>

<text
x="{width / 2}"
y="{height - 20}"
font-family="Arial"
font-size="14"
text-anchor="middle">
Test Period
</text>

<text
x="{left - 10}"
y="{top + 5}"
font-family="Arial"
font-size="12"
text-anchor="end">
{maximum:.1f}
</text>

<text
x="{left - 10}"
y="{height - bottom + 5}"
font-family="Arial"
font-size="12"
text-anchor="end">
{minimum:.1f}
</text>

</svg>"""

    output_path.write_text(
        svg,
        encoding="utf-8"
    )


create_svg_plot(
    prediction_df["Date"].tolist(),
    actual_sales,
    predicted_sales,
    PLOT_PATH
)


# --------------------------------------------------
# 32. Final generated files
# --------------------------------------------------

print("\nGenerated files:")

print(
    "LSTM model      :",
    MODEL_PATH
)

print(
    "Scaling data    :",
    SCALING_PATH
)

print(
    "Result summary  :",
    RESULTS_PATH
)

print(
    "Test predictions:",
    PREDICTIONS_PATH
)

print(
    "Last window     :",
    LAST_WINDOW_PATH
)

print(
    "Comparison plot :",
    PLOT_PATH
)