import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from models.ordinal_data import create_ordinal_datasets
from models.coral_loss import CoralLoss
from models.coral_utils import coral_predict, CLASS_NAMES

from config.config import MODEL_DIR, REPORT_DIR


print("=" * 70)
print("       NeuroCareAI 2.0 Evaluation Visualization")
print("=" * 70)


# ============================================================
# Load test dataset
# ============================================================

_, _, test_dataset = create_ordinal_datasets()


# ============================================================
# Load trained model
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

print("Model loaded successfully.")


# ============================================================
# Predictions
# ============================================================

y_true = []
y_pred = []


print("\nGenerating predictions...")


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
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


print("\nEvaluation Metrics")
print("-" * 50)

print(f"Accuracy          : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {precision * 100:.2f}%")
print(f"Macro Recall      : {recall * 100:.2f}%")
print(f"Macro F1 Score    : {f1 * 100:.2f}%")


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Save confusion matrix
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    values_format="d",
    xticks_rotation=45
)

ax.set_title(
    "NeuroCareAI 2.0 - Confusion Matrix"
)

plt.tight_layout()

confusion_path = os.path.join(
    REPORT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved:"
)

print(confusion_path)


# ============================================================
# Metrics chart
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

metric_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]


plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    metric_names,
    metric_values
)

plt.ylim(
    0,
    100
)

plt.ylabel(
    "Percentage (%)"
)

plt.title(
    "NeuroCareAI 2.0 Test Performance"
)

for bar, value in zip(
    bars,
    metric_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

metrics_path = os.path.join(
    REPORT_DIR,
    "evaluation_metrics.png"
)

plt.savefig(
    metrics_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nMetrics chart saved:"
)

print(metrics_path)


print("\n" + "=" * 70)
print("              VISUALIZATION COMPLETED")
print("=" * 70)