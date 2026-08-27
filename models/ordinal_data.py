import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

from config.config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED
)


# ============================================================
# Correct medical severity order
# ============================================================

ORDINAL_CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]


# TensorFlow alphabetical labels
TENSORFLOW_CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]


# Convert TensorFlow label -> ordinal label
LABEL_MAPPING = tf.constant(
    [2, 3, 0, 1],
    dtype=tf.int32
)


def convert_to_ordinal(images, labels):

    labels = tf.gather(
        LABEL_MAPPING,
        labels
    )

    return images, labels


def create_ordinal_datasets():

    print("Loading datasets...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Convert labels to medical ordinal order

    train_dataset = train_dataset.map(
        convert_to_ordinal,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    validation_dataset = validation_dataset.map(
        convert_to_ordinal,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_dataset = test_dataset.map(
        convert_to_ordinal,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    test_dataset = test_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


if __name__ == "__main__":

    train_ds, val_ds, test_ds = create_ordinal_datasets()

    print("\nOrdinal dataset created successfully.")

    print("\nCorrect class order:")

    for index, class_name in enumerate(
        ORDINAL_CLASS_NAMES
    ):
        print(f"{index} -> {class_name}")

    print("\nDataset sizes:")

    print(
        "Training batches:",
        tf.data.experimental.cardinality(
            train_ds
        ).numpy()
    )

    print(
        "Validation batches:",
        tf.data.experimental.cardinality(
            val_ds
        ).numpy()
    )

    print(
        "Testing batches:",
        tf.data.experimental.cardinality(
            test_ds
        ).numpy()
    )