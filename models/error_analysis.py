import os
import sys
import csv
import numpy as np
import tensorflow as tf

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# CONFIG
# ============================================================

from config.config import (
    TEST_DIR,
    MODEL_DIR,
    REPORT_DIR,
    IMAGE_SIZE,
    BATCH_SIZE
)

from models.coral_utils import (
    coral_predict,
    CLASS_NAMES
)

from models.ordinal_data import (
    convert_to_ordinal
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

CSV_PATH = os.path.join(
    REPORT_DIR,
    "error_analysis.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NeuroCareAI 2.0 ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(
    f"Model: {MODEL_PATH}"
)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATASET
#
# IMPORTANT:
# This uses EXACTLY the same TensorFlow dataset
# mechanism as ordinal_data.py / evaluation.
# ============================================================

print("\nLoading clean test dataset...")

raw_test_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
)


# Save file paths BEFORE map()
test_file_paths = list(
    raw_test_dataset.file_paths
)


# Convert TensorFlow alphabetical labels
# to medical ordinal labels

test_dataset = raw_test_dataset.map(
    convert_to_ordinal,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


print(
    f"Test images: {len(test_file_paths)}"
)


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning predictions...")

y_true = []
y_pred = []

errors = []

file_position = 0


for batch_number, (
    images,
    labels
) in enumerate(test_dataset):

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    logits = model(
        images,
        training=False
    )

    predictions, probabilities = (
        coral_predict(logits)
    )

    predicted_indices = (
        predictions.numpy()
    )

    true_indices = (
        labels.numpy()
    )

    probability_values = (
        probabilities.numpy()
    )


    # --------------------------------------------------------
    # Process batch
    # --------------------------------------------------------

    batch_size_actual = len(
        true_indices
    )

    batch_paths = test_file_paths[
        file_position:
        file_position + batch_size_actual
    ]


    for i in range(
        batch_size_actual
    ):

        actual_index = int(
            true_indices[i]
        )

        predicted_index = int(
            predicted_indices[i]
        )

        y_true.append(
            actual_index
        )

        y_pred.append(
            predicted_index
        )


        # ----------------------------------------------------
        # Misclassification
        # ----------------------------------------------------

        if (
            actual_index
            != predicted_index
        ):

            predicted_probability = float(
                probability_values[
                    i,
                    predicted_index
                ]
            )

            actual_probability = float(
                probability_values[
                    i,
                    actual_index
                ]
            )

            errors.append(
                {
                    "filename":
                        os.path.basename(
                            batch_paths[i]
                        ),

                    "image_path":
                        batch_paths[i],

                    "actual":
                        CLASS_NAMES[
                            actual_index
                        ],

                    "predicted":
                        CLASS_NAMES[
                            predicted_index
                        ],

                    "confidence":
                        predicted_probability,

                    "actual_probability":
                        actual_probability
                }
            )


    file_position += (
        batch_size_actual
    )


    if (
        (batch_number + 1) % 50 == 0
    ):

        print(
            f"Processed "
            f"{file_position}/"
            f"{len(test_file_paths)}"
        )


# ============================================================
# ARRAYS
# ============================================================

y_true = np.array(
    y_true,
    dtype=np.int32
)

y_pred = np.array(
    y_pred,
    dtype=np.int32
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = tf.math.confusion_matrix(
    y_true,
    y_pred,
    num_classes=len(
        CLASS_NAMES
    )
).numpy()


# ============================================================
# RESULTS
# ============================================================

total_images = len(
    y_true
)

correct = int(
    np.sum(
        y_true == y_pred
    )
)

incorrect = (
    total_images - correct
)

accuracy = (
    correct / total_images
)


print()
print("=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(
    f"Total test images : "
    f"{total_images}"
)

print(
    f"Correct           : "
    f"{correct}"
)

print(
    f"Misclassified     : "
    f"{incorrect}"
)

print(
    f"Accuracy          : "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix:")
print(confusion)


# ============================================================
# ERROR PAIRS
# ============================================================

pair_counts = {}


for error in errors:

    pair = (
        error["actual"],
        error["predicted"]
    )

    pair_counts[pair] = (
        pair_counts.get(
            pair,
            0
        ) + 1
    )


print()
print("Major error pairs:")

for (
    actual,
    predicted
), count in sorted(
    pair_counts.items(),
    key=lambda item: item[1],
    reverse=True
):

    print(
        f"{actual:20s} -> "
        f"{predicted:20s} : "
        f"{count}"
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "image_path",
            "actual",
            "predicted",
            "confidence",
            "actual_probability"
        ]
    )

    writer.writeheader()

    writer.writerows(
        errors
    )


print()
print(
    "Error report saved:"
)

print(
    CSV_PATH
)


# ============================================================
# TOP 50 ERRORS
# ============================================================

errors_sorted = sorted(
    errors,
    key=lambda item:
        item["confidence"],
    reverse=True
)


top_error_path = os.path.join(
    REPORT_DIR,
    "top_misclassified.txt"
)


with open(
    top_error_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "NeuroCareAI 2.0 "
        "TOP MISCLASSIFIED IMAGES\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    for number, error in enumerate(
        errors_sorted[:50],
        start=1
    ):

        file.write(
            f"{number}. "
            f"{error['filename']}\n"
        )

        file.write(
            f"   Actual      : "
            f"{error['actual']}\n"
        )

        file.write(
            f"   Prediction  : "
            f"{error['predicted']}\n"
        )

        file.write(
            f"   Confidence  : "
            f"{error['confidence'] * 100:.2f}%\n"
        )

        file.write(
            f"   Actual Prob : "
            f"{error['actual_probability'] * 100:.2f}%\n\n"
        )


print(
    "Top error report saved:"
)

print(
    top_error_path
)


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print()
print(
    "Expected accuracy from evaluation: "
    "96.16%"
)

print(
    f"Error-analysis accuracy: "
    f"{accuracy * 100:.2f}%"
)

if abs(
    accuracy - 0.9616
) < 0.001:

    print(
        "\nCONSISTENCY CHECK: PASSED"
    )

else:

    print(
        "\nCONSISTENCY CHECK: FAILED"
    )

    print(
        "Do NOT use this error analysis "
        "for the research paper yet."
    )