import numpy as np
import tensorflow as tf


def get_base_model(model):
    """
    Finds the nested MobileNetV2 model.
    """

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    raise ValueError(
        "No nested CNN model found."
    )


def get_last_conv_layer(base_model):
    """
    Finds the final usable convolutional layer.
    """

    for layer in reversed(base_model.layers):

        output_shape = getattr(
            layer,
            "output",
            None
        )

        if output_shape is not None:

            try:
                if len(layer.output.shape) == 4:
                    return layer
            except TypeError:
                pass

    raise ValueError(
        "No suitable convolutional layer found."
    )


def make_gradcam_heatmap(
    image_array,
    model
):
    """
    Creates a Grad-CAM heatmap.
    """

    base_model = get_base_model(model)

    last_conv_layer = get_last_conv_layer(
        base_model
    )

    # Model from MobileNet input
    # to final convolutional features
    conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # Find location of base model
    base_index = model.layers.index(
        base_model
    )

    # Create classifier working from
    # convolutional features onward
    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    # If final conv layer isn't the actual
    # final base layer, process remaining
    # base-model layers
    conv_index = base_model.layers.index(
        last_conv_layer
    )

    for layer in base_model.layers[
        conv_index + 1:
    ]:
        x = layer(x)

    # Run remaining outer classifier layers
    for layer in model.layers[
        base_index + 1:
    ]:
        x = layer(x)

    classifier_model = tf.keras.Model(
        classifier_input,
        x
    )

    # ------------------------------------------
    # Calculate gradients
    # ------------------------------------------

    with tf.GradientTape() as tape:

        conv_output = conv_model(
            image_array
        )

        tape.watch(
            conv_output
        )

        predictions = classifier_model(
            conv_output
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            :,
            predicted_class
        ]

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    # Average gradients
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    # Weight channels by importance
    heatmap = tf.reduce_sum(
        conv_output *
        pooled_gradients,
        axis=-1
    )

    # Remove negative contributions
    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    if maximum > 0:
        heatmap = heatmap / maximum

    return (
        heatmap.numpy(),
        int(predicted_class.numpy())
    )