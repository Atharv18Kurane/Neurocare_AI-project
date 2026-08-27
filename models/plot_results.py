import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf

from models.ordinal_data import create_ordinal_datasets
from models.coral_utils import coral_predict, CLASS_NAMES
from config.config import MODEL_DIR, REPORT_DIR


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

HISTORY_PATH = os.path.join(
    REPORT_DIR,
    "clean_training_history.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("NeuroCareAI 2.0 RESULT VISUALIZATION")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading clean test dataset...")

_, _, test_dataset = create_ordinal_datasets()

print("Test dataset loaded.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []


for images, labels in test_dataset:

    logits = model(
        images,
        training=False
    )

    predictions, _ = coral_predict(
        logits
    )

    y_true.extend(
        labels.numpy().tolist()
    )

    y_pred.extend(
        predictions.numpy().tolist()
    )


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

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(
        len(CLASS_NAMES)
    )
)


print("\nConfusion Matrix:")
print(cm)


fig, ax = plt.subplots(
    figsize=(9, 7)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "NeuroCareAI 2.0 - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()


cm_path = os.path.join(
    REPORT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nConfusion matrix saved:\n{cm_path}"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

if os.path.exists(HISTORY_PATH):

    print("\nLoading training history...")

    with open(
        HISTORY_PATH,
        "r"
    ) as f:

        history = json.load(f)


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    if (
        "loss" in history
        and "val_loss" in history
    ):

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            history["loss"],
            label="Training Loss"
        )

        plt.plot(
            history["val_loss"],
            label="Validation Loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "CORAL Loss"
        )

        plt.title(
            "Training and Validation Loss"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()


        loss_path = os.path.join(
            REPORT_DIR,
            "clean_training_loss.png"
        )

        plt.savefig(
            loss_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        print(
            f"Loss graph saved:\n{loss_path}"
        )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    if (
        "accuracy" in history
        and "val_accuracy" in history
    ):

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            history["accuracy"],
            label="Training Accuracy"
        )

        plt.plot(
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Accuracy"
        )

        plt.title(
            "Training and Validation Accuracy"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()


        accuracy_path = os.path.join(
            REPORT_DIR,
            "clean_training_accuracy.png"
        )

        plt.savefig(
            accuracy_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        print(
            f"Accuracy graph saved:\n{accuracy_path}"
        )


else:

    print(
        "\nTraining history file not found:"
    )

    print(HISTORY_PATH)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)