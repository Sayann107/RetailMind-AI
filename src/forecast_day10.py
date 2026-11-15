from pathlib import Path

import pandas as pd
import tensorflow as tf

from day10_attention import TemporalAttention


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

MODEL_PATH = Path(
    "models/day10_seq2seq.keras"
)

SCALING_PATH = Path(
    "models/day10_scaling.csv"
)

WINDOW_PATH = Path(
    "models/day10_last_window.csv"
)

FORECAST_PATH = Path(
    "models/day10_forecast.csv"
)


# --------------------------------------------------
# 2. Check files
# --------------------------------------------------

for file_path in [
    MODEL_PATH,
    SCALING_PATH,
    WINDOW_PATH
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing file: {file_path}\n"
            "Run train_day10_seq2seq.py first."
        )


# --------------------------------------------------
# 3. Load model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "TemporalAttention":
            TemporalAttention
    }
)


# --------------------------------------------------
# 4. Scaling details
# --------------------------------------------------

scaling = pd.read_csv(
    SCALING_PATH
).iloc[0]

store_id = int(
    scaling["Store"]
)

item_id = int(
    scaling["Item"]
)

look_back = int(
    scaling["Look Back"]
)

forecast_horizon = int(
    scaling["Forecast Horizon"]
)

sales_minimum = float(
    scaling["Minimum Sales"]
)

sales_maximum = float(
    scaling["Maximum Sales"]
)

sales_range = (
    sales_maximum -
    sales_minimum
)


# --------------------------------------------------
# 5. Load last history window
# --------------------------------------------------

window_df = pd.read_csv(
    WINDOW_PATH
)

window_df["date"] = pd.to_datetime(
    window_df["date"]
)

window = (
    window_df["scaled_sales"]
    .astype("float32")
    .tolist()
)


# --------------------------------------------------
# 6. Convert to tensor
# --------------------------------------------------

model_input = tf.convert_to_tensor(
    window,
    dtype=tf.float32
)

model_input = tf.reshape(
    model_input,
    (
        1,
        look_back,
        1
    )
)


# --------------------------------------------------
# 7. Predict all horizons
# --------------------------------------------------

scaled_forecast = model.predict(
    model_input,
    verbose=0
)[0]


# --------------------------------------------------
# 8. Convert to sales
# --------------------------------------------------

forecast_sales = []

for value in scaled_forecast:

    demand = (
        float(value) *
        sales_range +
        sales_minimum
    )

    forecast_sales.append(
        max(
            0.0,
            demand
        )
    )


# --------------------------------------------------
# 9. Future dates
# --------------------------------------------------

last_date = (
    window_df["date"].max()
)

future_dates = pd.date_range(
    start=(
        last_date +
        pd.Timedelta(days=1)
    ),
    periods=forecast_horizon,
    freq="D"
)


# --------------------------------------------------
# 10. Forecast table
# --------------------------------------------------

forecast_df = pd.DataFrame({
    "Date": future_dates,

    "Store": store_id,

    "Item": item_id,

    "Forecast Horizon": [
        f"Day +{index}"
        for index in range(
            1,
            forecast_horizon + 1
        )
    ],

    "Predicted Demand": [
        round(
            value,
            2
        )
        for value in forecast_sales
    ]
})


# --------------------------------------------------
# 11. Save
# --------------------------------------------------

forecast_df.to_csv(
    FORECAST_PATH,
    index=False
)


# --------------------------------------------------
# 12. Print
# --------------------------------------------------

print("\n" + "=" * 60)

print(
    "DAY 10 MULTI-HORIZON FORECAST"
)

print("=" * 60)

print(
    forecast_df.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    FORECAST_PATH
)