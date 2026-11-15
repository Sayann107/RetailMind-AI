from pathlib import Path

import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

FORECAST_DAYS = 7

MODEL_PATH = Path(
    "models/day9_lstm.keras"
)

SCALING_PATH = Path(
    "models/day9_scaling.csv"
)

LAST_WINDOW_PATH = Path(
    "models/day9_last_window.csv"
)

FORECAST_PATH = Path(
    "models/day9_forecast.csv"
)


# --------------------------------------------------
# 2. Check required files
# --------------------------------------------------

required_files = [
    MODEL_PATH,
    SCALING_PATH,
    LAST_WINDOW_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file_path}\n\n"
            "Run train_lstm_demand.py first."
        )


# --------------------------------------------------
# 3. Load model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


# --------------------------------------------------
# 4. Load scaling details
# --------------------------------------------------

scaling_df = pd.read_csv(
    SCALING_PATH
)

store_id = int(
    scaling_df.loc[
        0,
        "Store"
    ]
)

item_id = int(
    scaling_df.loc[
        0,
        "Item"
    ]
)

look_back = int(
    scaling_df.loc[
        0,
        "Look Back"
    ]
)

sales_minimum = float(
    scaling_df.loc[
        0,
        "Minimum Sales"
    ]
)

sales_maximum = float(
    scaling_df.loc[
        0,
        "Maximum Sales"
    ]
)

sales_range = (
    sales_maximum -
    sales_minimum
)


# --------------------------------------------------
# 5. Load latest sales window
# --------------------------------------------------

window_df = pd.read_csv(
    LAST_WINDOW_PATH
)

window_df["date"] = pd.to_datetime(
    window_df["date"]
)

current_window = (
    window_df["scaled_sales"]
    .astype("float32")
    .tolist()
)

if len(current_window) != look_back:

    raise ValueError(
        f"Expected {look_back} values, "
        f"but found {len(current_window)}."
    )


# --------------------------------------------------
# 6. Set first forecast date
# --------------------------------------------------

last_date = (
    window_df["date"]
    .max()
)

forecast_dates = pd.date_range(
    start=last_date +
    pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)


# --------------------------------------------------
# 7. Recursive forecast
# --------------------------------------------------

forecast_rows = []

for forecast_date in forecast_dates:

    model_input = tf.convert_to_tensor(
        current_window,
        dtype=tf.float32
    )

    model_input = tf.reshape(
        model_input,
        shape=(
            1,
            look_back,
            1
        )
    )

    scaled_forecast = float(
        model.predict(
            model_input,
            verbose=0
        )[0][0]
    )

    demand_forecast = (
        scaled_forecast *
        sales_range +
        sales_minimum
    )

    demand_forecast = max(
        0.0,
        demand_forecast
    )

    forecast_rows.append({
        "Date": forecast_date,
        "Store": store_id,
        "Item": item_id,
        "Predicted Demand":
            round(
                demand_forecast,
                2
            )
    })

    # Move the prediction into the window
    current_window.pop(0)

    current_window.append(
        scaled_forecast
    )


# --------------------------------------------------
# 8. Create forecast table
# --------------------------------------------------

forecast_df = pd.DataFrame(
    forecast_rows
)


# --------------------------------------------------
# 9. Save forecast
# --------------------------------------------------

forecast_df.to_csv(
    FORECAST_PATH,
    index=False
)


# --------------------------------------------------
# 10. Print forecast
# --------------------------------------------------

print("\n" + "=" * 60)
print("DAY 9: NEXT 7-DAY DEMAND FORECAST")
print("=" * 60)

print(
    forecast_df.to_string(
        index=False
    )
)

print(
    "\nForecast saved:",
    FORECAST_PATH
)