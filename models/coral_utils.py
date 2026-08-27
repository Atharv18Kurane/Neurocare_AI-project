import tensorflow as tf
import numpy as np


CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]


def coral_probabilities(logits):
    """
    Convert 3 CORAL logits into 4 mutually exclusive
    class probabilities.

    logits represent:
        P(y > 0)
        P(y > 1)
        P(y > 2)
    """

    cumulative = tf.sigmoid(logits)

    p_gt_0 = cumulative[:, 0]
    p_gt_1 = cumulative[:, 1]
    p_gt_2 = cumulative[:, 2]

    # Four class probabilities
    p_class_0 = 1.0 - p_gt_0

    p_class_1 = p_gt_0 - p_gt_1

    p_class_2 = p_gt_1 - p_gt_2

    p_class_3 = p_gt_2

    probabilities = tf.stack(
        [
            p_class_0,
            p_class_1,
            p_class_2,
            p_class_3
        ],
        axis=1
    )

    # Numerical safety
    probabilities = tf.clip_by_value(
        probabilities,
        0.0,
        1.0
    )

    # Normalize
    probabilities = probabilities / (
        tf.reduce_sum(
            probabilities,
            axis=1,
            keepdims=True
        ) + 1e-8
    )

    return probabilities


def coral_predict(logits):
    """
    Convert CORAL logits into class predictions.
    """

    probabilities = coral_probabilities(logits)

    predictions = tf.argmax(
        probabilities,
        axis=1
    )

    return predictions, probabilities


def decode_class(class_index):

    return CLASS_NAMES[int(class_index)]