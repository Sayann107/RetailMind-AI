from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

from gradcam import make_gradcam_heatmap


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

IMAGE_SIZE = 160

MODEL_PATH = Path(
    "models/fashion_transfer_model.keras"
)

CLASS_NAMES_PATH = Path(
    "models/day7_class_names.txt"
)

OUTPUT_DIRECTORY = Path(
    "outputs/day8_gradcam"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Check Day 7 model
# --------------------------------------------------

if not MODEL_PATH.exists():

    raise FileNotFoundError(
    "\nDay 7 model not found:\n"
    "models/fashion_transfer_model.keras\n\n"
    "Run train_transfer_learning.py first."
)


if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(
        "\nDay 7 class names not found:\n"
        "models/day7_class_names.txt"
    )


# --------------------------------------------------
# 3. Load Day 7 model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


# --------------------------------------------------
# 4. Load categories
# --------------------------------------------------

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]


# --------------------------------------------------
# 5. Explain one product
# --------------------------------------------------

def explain_product(image_path):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # ------------------------------------------
    # Load product image
    # ------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )

    original_image = (
        image_array / 255.0
    )


    # ------------------------------------------
    # Prepare model input
    # ------------------------------------------

    model_input = np.expand_dims(
        image_array,
        axis=0
    )

    model_input = preprocess_input(
        model_input
    )


    # ------------------------------------------
    # Product prediction
    # ------------------------------------------

    probabilities = model.predict(
        model_input,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_category = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    # ------------------------------------------
    # Grad-CAM
    # ------------------------------------------

    heatmap, _ = make_gradcam_heatmap(
        model_input,
        model
    )


    # Resize heatmap
    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ).numpy()

    heatmap = np.squeeze(
        heatmap
    )


    # ------------------------------------------
    # Plot explanation
    # ------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )


    # Original
    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        "Original Product"
    )

    plt.axis(
        "off"
    )


    # Grad-CAM
    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        original_image
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.4
    )

    plt.title(
        f"{predicted_category}\n"
        f"Confidence: {confidence:.2%}"
    )

    plt.axis(
        "off"
    )


    # ------------------------------------------
    # Save explanation
    # ------------------------------------------

    output_file = (
        OUTPUT_DIRECTORY /
        f"{image_path.stem}_gradcam.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()


    # ------------------------------------------
    # Return explanation
    # ------------------------------------------

    return {
        "category":
            predicted_category,

        "confidence":
            round(confidence, 4),

        "gradcam":
            str(output_file)
    }


# --------------------------------------------------
# 6. Run from terminal
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            "python src/explain_product.py "
            "\"path/to/image.jpg\""
        )

        raise SystemExit(1)


    result = explain_product(
        sys.argv[1]
    )


    print("\n" + "=" * 50)

    print(
        "DAY 8 GRAD-CAM EXPLANATION"
    )

    print("=" * 50)

    print(
        "Predicted category:",
        result["category"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print(
        "Explanation:",
        result["gradcam"]
    )