from pathlib import Path
import shutil
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CSV_PATH = Path(
    "data/myntradataset/styles.csv"
)

IMAGE_DIRECTORY = Path(
    "data/myntradataset/images"
)

OUTPUT_DIRECTORY = Path(
    "data/day11_rare_category"
)

CATEGORY_FILE = Path(
    "models/day11_category.txt"
)

MAX_IMAGES = 200


# Same major categories used in Day 7
CATEGORIES = [
    "Apparel",
    "Footwear",
    "Accessories",
    "Personal Care"
]


# --------------------------------------------------
# Check paths
# --------------------------------------------------

if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"styles.csv not found: {CSV_PATH}"
    )


if not IMAGE_DIRECTORY.exists():

    raise FileNotFoundError(
        f"Image directory not found: "
        f"{IMAGE_DIRECTORY}"
    )


CATEGORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

df = pd.read_csv(
    CSV_PATH,
    on_bad_lines="skip"
)


required_columns = [
    "id",
    "masterCategory"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


# --------------------------------------------------
# Filter categories
# --------------------------------------------------

df = df[
    df["masterCategory"].isin(
        CATEGORIES
    )
].copy()


counts = (
    df["masterCategory"]
    .value_counts()
)


print("\nCATEGORY COUNTS")
print(counts)


# --------------------------------------------------
# Find the smallest category
# --------------------------------------------------

rare_category = counts.idxmin()


print(
    "\nSelected under-represented category:",
    rare_category
)


# --------------------------------------------------
# Prepare folder
# --------------------------------------------------

if OUTPUT_DIRECTORY.exists():

    shutil.rmtree(
        OUTPUT_DIRECTORY
    )


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Find images belonging to selected category
# --------------------------------------------------

category_rows = df[
    df["masterCategory"] ==
    rare_category
].sample(
    frac=1,
    random_state=42
)


copied = 0


for _, row in category_rows.iterrows():

    image_id = str(
        int(row["id"])
    )

    source = (
        IMAGE_DIRECTORY /
        f"{image_id}.jpg"
    )


    if not source.exists():
        continue


    destination = (
        OUTPUT_DIRECTORY /
        source.name
    )


    shutil.copy2(
        source,
        destination
    )


    copied += 1


    if copied >= MAX_IMAGES:
        break


# --------------------------------------------------
# Save category name
# --------------------------------------------------

CATEGORY_FILE.write_text(
    rare_category,
    encoding="utf-8"
)


print(
    "\nImages copied:",
    copied
)

print(
    "Prepared directory:",
    OUTPUT_DIRECTORY
)

print(
    "Category saved:",
    CATEGORY_FILE
)