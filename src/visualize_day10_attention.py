from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "models/day10_attention_weights.csv"
)

OUTPUT_PATH = Path(
    "outputs/day10_attention.svg"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


if not INPUT_PATH.exists():

    raise FileNotFoundError(
        "Attention weights not found.\n"
        "Run train_day10_seq2seq.py first."
    )


df = pd.read_csv(
    INPUT_PATH
)


width = 1000
height = 500

left = 80
right = 30
top = 60
bottom = 90

plot_width = (
    width -
    left -
    right
)

plot_height = (
    height -
    top -
    bottom
)

maximum_weight = df[
    "Attention Weight"
].max()

if maximum_weight <= 0:
    maximum_weight = 1


bar_width = (
    plot_width /
    len(df)
)


bars = []

for index, row in df.iterrows():

    value = float(
        row["Attention Weight"]
    )

    bar_height = (
        value /
        maximum_weight *
        plot_height
    )

    x = (
        left +
        index *
        bar_width
    )

    y = (
        top +
        plot_height -
        bar_height
    )

    bars.append(
        f"""
        <rect
        x="{x:.2f}"
        y="{y:.2f}"
        width="{max(bar_width - 2, 1):.2f}"
        height="{bar_height:.2f}"
        fill="#7C3AED"/>
        """
    )


svg = f"""
<svg
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
y="30"
text-anchor="middle"
font-family="Arial"
font-size="20">
Day 10: Attention Across 30-Day History
</text>

<line
x1="{left}"
y1="{top}"
x2="{left}"
y2="{top + plot_height}"
stroke="black"/>

<line
x1="{left}"
y1="{top + plot_height}"
x2="{width - right}"
y2="{top + plot_height}"
stroke="black"/>

{''.join(bars)}

<text
x="{width / 2}"
y="{height - 30}"
text-anchor="middle"
font-family="Arial"
font-size="14">
Past Days
</text>

<text
x="25"
y="{height / 2}"
text-anchor="middle"
font-family="Arial"
font-size="14"
transform="rotate(-90 25 {height / 2})">
Attention Weight
</text>

</svg>
"""


OUTPUT_PATH.write_text(
    svg,
    encoding="utf-8"
)

print(
    "Attention visualization saved:",
    OUTPUT_PATH
)