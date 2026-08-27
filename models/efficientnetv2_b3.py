import os
import sys

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

from config.config import (
    IMAGE_SIZE,
    CORAL_OUTPUTS
)

from models.coral_loss import CoralLoss


def build_efficientnetv2_b3(
    trainable_base=False
):

    # ========================================================
    # Input
    # ========================================================

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        name="mri_input"
    )

    # ========================================================
    # Data augmentation
    # ========================================================

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                0.03
            ),

            tf.keras.layers.RandomZoom(
                0.10
            ),

            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05
            ),

            tf.keras.layers.RandomContrast(
                0.10
            )
        ],
        name="data_augmentation"
    )

    x = augmentation(inputs)

    # ========================================================
    # EfficientNetV2-B3
    # ========================================================

    base_model = tf.keras.applications.EfficientNetV2B3(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        include_preprocessing=True
    )

    base_model.trainable = trainable_base

    x = base_model(
        x,
        training=trainable_base
    )

    # ========================================================
    # Feature extraction
    # ========================================================

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="feature_batch_normalization"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout"
    )(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        name="feature_dense"
    )(x)

    x = tf.keras.layers.Dropout(
        0.20,
        name="classifier_dropout"
    )(x)

    # ========================================================
    # CORAL ordinal output
    # ========================================================

    outputs = tf.keras.layers.Dense(
        CORAL_OUTPUTS,
        activation=None,
        name="coral_logits"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="NeuroCareAI_EfficientNetV2B3_CORAL"
    )

    return model


if __name__ == "__main__":

    model = build_efficientnetv2_b3(
        trainable_base=False
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=CoralLoss()
    )

    model.summary()