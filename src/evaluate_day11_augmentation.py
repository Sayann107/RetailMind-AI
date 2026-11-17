from pathlib import Path
import shutil
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

REAL_DIRECTORY = Path(
    "data/day11_rare_category"
)


SYNTHETIC_DIRECTORY = Path(
    "outputs/day11_synthetic"
)


CATEGORY_PATH = Path(
    "models/day11_category.txt"
)


AUGMENTED_DIRECTORY = Path(
    "data/day11_augmented_category"
)


RESULTS_PATH = Path(
    "models/day11_results.csv"
)


# --------------------------------------------------
# Check inputs
# --------------------------------------------------

if not REAL_DIRECTORY.exists():

    raise FileNotFoundError(
        "Real Day 11 image folder missing."
    )


if not SYNTHETIC_DIRECTORY.exists():

    raise FileNotFoundError(
        "Synthetic image folder missing."
    )


if not CATEGORY_PATH.exists():

    raise FileNotFoundError(
        "Day 11 category file missing."
    )


# --------------------------------------------------
# Category
# --------------------------------------------------

category = (
    CATEGORY_PATH
    .read_text(
        encoding="utf-8"
    )
    .strip()
)


# --------------------------------------------------
# Count images
# --------------------------------------------------

real_images = list(
    REAL_DIRECTORY.glob("*.jpg")
)


synthetic_images = list(
    SYNTHETIC_DIRECTORY.glob("*.png")
)


# --------------------------------------------------
# Rebuild augmented dataset
# --------------------------------------------------

if AUGMENTED_DIRECTORY.exists():

    shutil.rmtree(
        AUGMENTED_DIRECTORY
    )


AUGMENTED_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Copy real images
# --------------------------------------------------

for image in real_images:

    shutil.copy2(
        image,
        AUGMENTED_DIRECTORY /
        image.name
    )


# --------------------------------------------------
# Copy approved synthetic images
# --------------------------------------------------

for image in synthetic_images:

    shutil.copy2(
        image,
        AUGMENTED_DIRECTORY /
        image.name
    )


# --------------------------------------------------
# Save result
# --------------------------------------------------

results = pd.DataFrame([
    {
        "Category":
            category,

        "Real Images":
            len(real_images),

        "Synthetic Images Kept":
            len(synthetic_images),

        "Augmented Total":
            (
                len(real_images) +
                len(synthetic_images)
            )
    }
])


results.to_csv(
    RESULTS_PATH,
    index=False
)


print("\nDAY 11 AUGMENTATION SUMMARY")

print(
    results.to_string(
        index=False
    )
)


print(
    "\nPrepared augmented category:",
    AUGMENTED_DIRECTORY
)


print(
    "Result saved:",
    RESULTS_PATH
)