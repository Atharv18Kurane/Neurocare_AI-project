import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="NeuroCareAI"
)
class CoralLoss(tf.keras.losses.Loss):

    """
    CORAL ordinal classification loss.

    Class order:

        0 = NonDemented
        1 = VeryMildDemented
        2 = MildDemented
        3 = ModerateDemented

    CORAL outputs:

        P(y > 0)
        P(y > 1)
        P(y > 2)
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="coral_loss"
    ):

        super().__init__(
            reduction=reduction,
            name=name
        )

    def call(self, y_true, y_pred):

        y_true = tf.cast(
            tf.reshape(y_true, [-1]),
            tf.int32
        )

        num_thresholds = tf.shape(y_pred)[1]

        thresholds = tf.range(
            num_thresholds,
            dtype=tf.int32
        )

        ordinal_targets = tf.cast(
            tf.expand_dims(y_true, axis=1) > thresholds,
            tf.float32
        )

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ordinal_targets,
            logits=y_pred
        )

        return tf.reduce_mean(loss)

    def get_config(self):

        config = super().get_config()

        return config