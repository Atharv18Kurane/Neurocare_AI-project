import os
import sys


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

from explainability.gradcam import save_gradcam


# ============================================================
# EXACT NON-DEMENTED TEST IMAGE
# ============================================================

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw",
    "NonDemented",
    "00e3c81d-5035-42d7-93f3-09a5a84074c2.jpg"
)


# ============================================================
# CHECK IMAGE
# ============================================================

print("=" * 60)
print("NeuroCareAI 2.0 Grad-CAM Test")
print("=" * 60)

print("\nTesting image:")
print(IMAGE_PATH)


if not os.path.exists(IMAGE_PATH):

    print("\nERROR: Image does not exist.")

    print("\nChecking directory:")

    directory = os.path.dirname(
        IMAGE_PATH
    )

    print(directory)

    if os.path.exists(directory):

        print("\nFiles found:")

        for filename in os.listdir(directory):

            print(filename)

    else:

        print("Directory does not exist.")

    raise FileNotFoundError(
        f"\nMRI image not found:\n{IMAGE_PATH}"
    )


print("\nImage found successfully.")


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

prediction, confidence, probabilities, output_path = save_gradcam(
    IMAGE_PATH
)


# ============================================================
# RESULT
# ============================================================

print("\n")
print("=" * 60)
print("GRAD-CAM TEST RESULT")
print("=" * 60)

print(
    f"Prediction: {prediction}"
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)


print("\nProbabilities:")


# Handle dictionary or numpy/TensorFlow output

if hasattr(
    probabilities,
    "items"
):

    for class_name, probability in probabilities.items():

        print(
            f"{class_name}: "
            f"{float(probability) * 100:.2f}%"
        )

else:

    import numpy as np

    probability_array = np.asarray(
        probabilities
    ).reshape(-1)

    from models.coral_utils import CLASS_NAMES

    for index, probability in enumerate(
        probability_array
    ):

        print(
            f"{index} - "
            f"{CLASS_NAMES[index]}: "
            f"{float(probability) * 100:.2f}%"
        )


print("\nGrad-CAM saved:")
print(output_path)

print("=" * 60)