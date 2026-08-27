import os
import sys
import random

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT
# ============================================================

from models.predict import predict_image


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw",
    "NonDemented"
)

CLASS_NAME = "NonDemented"

NUM_IMAGES = 10


DATASET_DIR = r"dataset\raw\NonDemented"

CLASS_NAME = "NonDemented"

NUM_IMAGES = 10


files = [
    f
    for f in os.listdir(DATASET_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
]


random.seed(42)

selected_files = random.sample(
    files,
    NUM_IMAGES
)


correct = 0


print("=" * 70)
print("NON-DEMENTED ERROR ANALYSIS")
print("=" * 70)


for index, filename in enumerate(
    selected_files,
    start=1
):

    image_path = os.path.join(
        DATASET_DIR,
        filename
    )

    prediction, confidence, probabilities = predict_image(
        image_path
    )

    if prediction == CLASS_NAME:
        correct += 1

    print(
        f"\n{index}. {filename}"
    )

    print(
        f"   Actual      : {CLASS_NAME}"
    )

    print(
        f"   Prediction  : {prediction}"
    )

    print(
        f"   Confidence  : {confidence * 100:.2f}%"
    )

    print(
        "   Probabilities:"
    )

    for class_name, probability in probabilities.items():

        print(
            f"      {class_name:<20}"
            f"{probability * 100:>7.2f}%"
        )


print("\n" + "=" * 70)

print(
    f"Correct: {correct}/{NUM_IMAGES}"
)

print(
    f"Accuracy: {correct / NUM_IMAGES * 100:.2f}%"
)

print("=" * 70)