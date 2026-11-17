from pathlib import Path

import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

LATENT_DIM = 32

NUMBER_OF_IMAGES = 20


DECODER_PATH = Path(
    "models/day11_vae_decoder.keras"
)


OUTPUT_DIRECTORY = Path(
    "outputs/day11_synthetic"
)


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Check model
# --------------------------------------------------

if not DECODER_PATH.exists():

    raise FileNotFoundError(
        "VAE decoder not found.\n"
        "Run train_day11_vae.py first."
    )


# --------------------------------------------------
# Load decoder
# --------------------------------------------------

decoder = tf.keras.models.load_model(
    DECODER_PATH
)


# --------------------------------------------------
# Generate random latent vectors
# --------------------------------------------------

tf.keras.utils.set_random_seed(
    42
)


noise = tf.random.normal(
    shape=(
        NUMBER_OF_IMAGES,
        LATENT_DIM
    )
)


generated_images = decoder.predict(
    noise,
    verbose=0
)


# --------------------------------------------------
# Save generated products
# --------------------------------------------------

for index in range(
    NUMBER_OF_IMAGES
):

    image = generated_images[
        index
    ]


    image = tf.clip_by_value(
        image,
        0.0,
        1.0
    )


    image = tf.image.convert_image_dtype(
        image,
        dtype=tf.uint8,
        saturate=True
    )


    output_path = (
        OUTPUT_DIRECTORY /
        f"synthetic_{index + 1:03d}.png"
    )


    encoded_image = (
        tf.io.encode_png(
            image
        )
    )


    tf.io.write_file(
        str(output_path),
        encoded_image
    )


print(
    "\nGenerated images:",
    NUMBER_OF_IMAGES
)


print(
    "Saved in:",
    OUTPUT_DIRECTORY
)