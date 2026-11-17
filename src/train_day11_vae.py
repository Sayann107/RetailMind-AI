from pathlib import Path
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Reshape,
    Lambda
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_SIZE = 64

BATCH_SIZE = 16

EPOCHS = 15

LATENT_DIM = 32


DATA_DIRECTORY = Path(
    "data/day11_rare_category"
)

MODEL_DIRECTORY = Path(
    "models"
)

ENCODER_PATH = (
    MODEL_DIRECTORY /
    "day11_vae_encoder.keras"
)

DECODER_PATH = (
    MODEL_DIRECTORY /
    "day11_vae_decoder.keras"
)


MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


tf.keras.utils.set_random_seed(
    42
)


# --------------------------------------------------
# Check images
# --------------------------------------------------

if not DATA_DIRECTORY.exists():

    raise FileNotFoundError(
        "Day 11 images not found.\n"
        "Run prepare_day11_category.py first."
    )


image_files = list(
    DATA_DIRECTORY.glob("*.jpg")
)


if len(image_files) < 10:

    raise ValueError(
        "Not enough images to train VAE."
    )


print(
    "Training images:",
    len(image_files)
)


# --------------------------------------------------
# Load images
# --------------------------------------------------

def load_image(path):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    return image


dataset = tf.data.Dataset.from_tensor_slices(
    [
        str(path)
        for path in image_files
    ]
)


dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


dataset = dataset.shuffle(
    len(image_files),
    seed=42
)


dataset = dataset.batch(
    BATCH_SIZE
)


dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)


# --------------------------------------------------
# Sampling function
# --------------------------------------------------

def sampling(args):

    z_mean, z_log_var = args

    batch = tf.shape(
        z_mean
    )[0]

    dimension = tf.shape(
        z_mean
    )[1]

    epsilon = tf.random.normal(
        shape=(
            batch,
            dimension
        )
    )

    return (
        z_mean +
        tf.exp(
            0.5 *
            z_log_var
        ) *
        epsilon
    )


# --------------------------------------------------
# Encoder
# --------------------------------------------------

encoder_input = Input(
    shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)


x = Conv2D(
    32,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(encoder_input)


x = Conv2D(
    64,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)


x = Conv2D(
    128,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)


x = Flatten()(x)


x = Dense(
    128,
    activation="relu"
)(x)


z_mean = Dense(
    LATENT_DIM,
    name="z_mean"
)(x)


z_log_var = Dense(
    LATENT_DIM,
    name="z_log_var"
)(x)


z = Lambda(
    sampling,
    name="sampling"
)(
    [
        z_mean,
        z_log_var
    ]
)


encoder = Model(
    encoder_input,
    [
        z_mean,
        z_log_var,
        z
    ],
    name="day11_encoder"
)


# --------------------------------------------------
# Decoder
# --------------------------------------------------

decoder_input = Input(
    shape=(
        LATENT_DIM,
    )
)


x = Dense(
    8 * 8 * 128,
    activation="relu"
)(decoder_input)


x = Reshape(
    (
        8,
        8,
        128
    )
)(x)


x = Conv2DTranspose(
    128,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)


x = Conv2DTranspose(
    64,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)


x = Conv2DTranspose(
    32,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)


decoder_output = Conv2D(
    3,
    3,
    padding="same",
    activation="sigmoid"
)(x)


decoder = Model(
    decoder_input,
    decoder_output,
    name="day11_decoder"
)


# --------------------------------------------------
# VAE model
# --------------------------------------------------

class VAE(Model):

    def __init__(
        self,
        encoder,
        decoder
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


        self.total_loss_tracker = (
            tf.keras.metrics.Mean(
                name="loss"
            )
        )


    @property
    def metrics(self):

        return [
            self.total_loss_tracker
        ]


    def train_step(
        self,
        data
    ):

        with tf.GradientTape() as tape:

            (
                z_mean,
                z_log_var,
                z
            ) = self.encoder(
                data,
                training=True
            )


            reconstruction = (
                self.decoder(
                    z,
                    training=True
                )
            )


            reconstruction_loss = (
                tf.reduce_mean(
                    tf.reduce_sum(
                        tf.square(
                            data -
                            reconstruction
                        ),
                        axis=(
                            1,
                            2,
                            3
                        )
                    )
                )
            )


            kl_loss = (
                -0.5 *
                tf.reduce_mean(
                    tf.reduce_sum(
                        1 +
                        z_log_var -
                        tf.square(
                            z_mean
                        ) -
                        tf.exp(
                            z_log_var
                        ),
                        axis=1
                    )
                )
            )


            total_loss = (
                reconstruction_loss +
                0.001 *
                kl_loss
            )


        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )


        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_weights
            )
        )


        self.total_loss_tracker.update_state(
            total_loss
        )


        return {
            "loss":
                self.total_loss_tracker.result()
        }


# --------------------------------------------------
# Train
# --------------------------------------------------

vae = VAE(
    encoder,
    decoder
)


vae.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    )
)


print("\nEncoder:")
encoder.summary()


print("\nDecoder:")
decoder.summary()


vae.fit(
    dataset,
    epochs=EPOCHS,
    verbose=1
)


# --------------------------------------------------
# Save models
# --------------------------------------------------

encoder.save(
    ENCODER_PATH
)


decoder.save(
    DECODER_PATH
)


print(
    "\nEncoder saved:",
    ENCODER_PATH
)


print(
    "Decoder saved:",
    DECODER_PATH
)