import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="RetailMind"
)
class TemporalAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.score_layer = tf.keras.layers.Dense(
            1
        )

    def call(self, inputs):

        scores = self.score_layer(
            inputs
        )

        weights = tf.nn.softmax(
            scores,
            axis=1
        )

        context = tf.reduce_sum(
            inputs * weights,
            axis=1
        )

        return context, weights

    def get_config(self):
        return super().get_config()