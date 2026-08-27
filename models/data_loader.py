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
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED
)


# ============================================================
# TensorFlow class order
# ============================================================

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]


# ============================================================
# Create datasets
# ============================================================

def create_datasets():

    print("Loading training dataset...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    print("Loading validation dataset...")

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Loading test dataset...")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Save class names BEFORE prefetch
    class_names = train_dataset.class_names

    print("\nTensorFlow class names:")
    print(class_names)

    # ========================================================
    # Performance optimization
    # ========================================================

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    train_ds, val_ds, test_ds, class_names = create_datasets()

    print("\nDataset loading successful!")

    print("\nClass names:")
    print(class_names)

    print(
        "\nNumber of training batches:",
        tf.data.experimental.cardinality(train_ds).numpy()
    )

    print(
        "Number of validation batches:",
        tf.data.experimental.cardinality(val_ds).numpy()
    )

    print(
        "Number of testing batches:",
        tf.data.experimental.cardinality(test_ds).numpy()
    )