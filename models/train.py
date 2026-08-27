import os
import sys

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import tensorflow as tf
import matplotlib.pyplot as plt

from models.efficientnetv2_b3 import build_efficientnetv2_b3
from models.ordinal_data import create_ordinal_datasets
from models.coral_loss import CoralLoss

from config.config import (
    MODEL_DIR,
    REPORT_DIR,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE
)


# ============================================================
# GPU / CPU information
# ============================================================

print("=" * 70)
print("              NeuroCareAI 2.0 Training")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("\nWARNING: No GPU detected.")
    print("Training will run on CPU.")


# ============================================================
# Load datasets
# ============================================================

print("\nLoading datasets...")

train_dataset, validation_dataset, test_dataset = (
    create_ordinal_datasets()
)

print("\nDatasets loaded successfully.")


# ============================================================
# Build model
# ============================================================

print("\nBuilding EfficientNetV2-B3 model...")

model = build_efficientnetv2_b3(
    trainable_base=False
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),
    loss=CoralLoss()
)

print("\nModel created successfully.")

model.summary()


# ============================================================
# Callbacks - Stage 1
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

stage1_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# Stage 1 Training
# ============================================================

print("\n")
print("=" * 70)
print("                 STAGE 1 TRAINING")
print("=" * 70)

history_stage1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        stage1_checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# Stage 2 - Fine tuning
# ============================================================

print("\n")
print("=" * 70)
print("                 STAGE 2 FINE-TUNING")
print("=" * 70)


# Find EfficientNet backbone
base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ) and "efficientnetv2" in layer.name.lower():

        base_model = layer
        break


if base_model is None:

    raise RuntimeError(
        "EfficientNetV2-B3 backbone not found."
    )


print(
    "\nBackbone found:",
    base_model.name
)


# ------------------------------------------------------------
# Freeze everything first
# ------------------------------------------------------------

base_model.trainable = True


# ------------------------------------------------------------
# Freeze lower layers
# ------------------------------------------------------------

fine_tune_at = int(
    len(base_model.layers) * 0.70
)

print(
    "Total backbone layers:",
    len(base_model.layers)
)

print(
    "Fine-tuning from layer:",
    fine_tune_at
)


for layer in base_model.layers[:fine_tune_at]:

    layer.trainable = False


# ------------------------------------------------------------
# Keep BatchNormalization layers frozen
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# Recompile after changing trainable layers
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LEARNING_RATE
    ),
    loss=CoralLoss()
)


# ============================================================
# Stage 2 callbacks
# ============================================================

stage2_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

stage2_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

stage2_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# Stage 2 Training
# ============================================================

history_stage2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    initial_epoch=INITIAL_EPOCHS,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=[
        stage2_checkpoint,
        stage2_early_stopping,
        stage2_reduce_lr
    ]
)


# ============================================================
# Save final model
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)

model.save(final_model_path)

print("\nFinal model saved:")
print(final_model_path)

print("\nBest model saved:")
print(best_model_path)


# ============================================================
# Combine training histories
# ============================================================

stage1_loss = history_stage1.history["loss"]
stage1_val_loss = history_stage1.history["val_loss"]

stage2_loss = history_stage2.history["loss"]
stage2_val_loss = history_stage2.history["val_loss"]


all_loss = stage1_loss + stage2_loss
all_val_loss = stage1_val_loss + stage2_val_loss


# ============================================================
# Plot loss
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    all_loss,
    label="Training Loss"
)

plt.plot(
    all_val_loss,
    label="Validation Loss"
)

plt.axvline(
    x=len(stage1_loss) - 1,
    linestyle="--",
    label="Fine-tuning Start"
)

plt.xlabel("Epoch")
plt.ylabel("CORAL Loss")
plt.title(
    "NeuroCareAI 2.0 Training and Validation Loss"
)

plt.legend()
plt.grid(True)

loss_path = os.path.join(
    REPORT_DIR,
    "loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nLoss graph saved:")
print(loss_path)


print("\n")
print("=" * 70)
print("             TRAINING COMPLETED")
print("=" * 70)