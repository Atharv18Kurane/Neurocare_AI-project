import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from PIL import Image

from config.config import (
    IMAGE_SIZE,
    MODEL_DIR
)

from models.coral_utils import (
    coral_predict,
    CLASS_NAMES
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)


# Load model once
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def predict_image(image_path):

    image = preprocess_image(
        image_path
    )

    logits = model(
        image,
        training=False
    )

    predicted_class, probabilities = coral_predict(
        logits
    )

    predicted_index = int(
        predicted_class.numpy()[0]
    )

    probability_values = (
        probabilities.numpy()[0]
    )

    prediction = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probability_values[
            predicted_index
        ]
    )

    class_probabilities = {
        CLASS_NAMES[i]: float(
            probability_values[i]
        )
        for i in range(
            len(CLASS_NAMES)
        )
    }

    return (
        prediction,
        confidence,
        class_probabilities
    )


if __name__ == "__main__":

    print("NeuroCareAI 2.0 prediction module")
    print("Model loaded successfully.")
    print("\nClasses:")

    for index, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{index} -> {name}"
        )