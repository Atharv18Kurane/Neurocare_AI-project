import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error
)

from models.ordinal_data import create_ordinal_datasets
from models.coral_loss import CoralLoss
from models.coral_utils import (
    coral_predict,
    CLASS_NAMES
)

from config.config import MODEL_DIR, REPORT_DIR


# ============================================================
# Load test dataset
# ============================================================

print("=" * 70)
print("              NeuroCareAI 2.0 Evaluation")
print("=" * 70)

_, _, test_dataset = create_ordinal_datasets()


# ============================================================
# Load best model
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

print("\nLoading model:")
print(model_path)

model = tf.keras.models.load_model(
    model_path,
    compile=False
)


print("\nModel loaded successfully.")


# ============================================================
# Predictions
# ============================================================

y_true = []
y_pred = []


print("\nRunning test prediction...")


for images, labels in test_dataset:

    logits = model(
        images,
        training=False
    )

    predictions, probabilities = coral_predict(
        logits
    )

    y_true.extend(
        labels.numpy().tolist()
    )

    y_pred.extend(
        predictions.numpy().tolist()
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Accuracy
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n")
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Ordinal Mean Absolute Error
# ============================================================

mae = mean_absolute_error(
    y_true,
    y_pred
)

print(
    f"\nOrdinal MAE: {mae:.4f}"
)


# ============================================================
# Save report
# ============================================================

report_path = os.path.join(
    REPORT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "NeuroCareAI 2.0 Evaluation Report\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Ordinal MAE: {mae:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(report)

    file.write(
        "\n\nConfusion Matrix\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        np.array2string(cm)
    )


print("\nReport saved:")
print(report_path)

print("\nEvaluation completed.")