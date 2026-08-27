import os
import sys
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
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    MODEL_DIR,
    REPORT_DIR
)


# ============================================================
# DATA
# ============================================================

from models.ordinal_data import (
    create_ordinal_datasets
)


# ============================================================
# MODEL
# ============================================================

from models.efficientnetv2_b3 import (
    build_efficientnetv2_b3
)

from models.coral_loss import (
    CoralLoss
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("NeuroCareAI 2.0")
print("CLEAN DATASET TRAINING")
print("=" * 70)

print()
print("Dataset:")
print("  dataset/clean")
print()

print("Model:")
print("  EfficientNetV2-B3")
print()

print("Loss:")
print("  CORAL")
print()

print("Training:")
print(
    f"  Stage 1 epochs : {INITIAL_EPOCHS}"
)

print(
    f"  Stage 2 epochs : {FINE_TUNE_EPOCHS}"
)

print(
    f"  Total epochs   : "
    f"{INITIAL_EPOCHS + FINE_TUNE_EPOCHS}"
)

print()


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING CLEAN DATASET")
print("=" * 70)

(
    train_dataset,
    validation_dataset,
    test_dataset
) = create_ordinal_datasets()


print()
print("Clean dataset loaded successfully.")


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 70)
print("BUILDING EFFICIENTNETV2-B3")
print("=" * 70)

model = build_efficientnetv2_b3(
    trainable_base=False
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),
    loss=CoralLoss()
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# STAGE 1
# ============================================================

print()
print("=" * 70)
print("STAGE 1 TRAINING")
print("=" * 70)

print(
    "EfficientNetV2-B3 backbone: FROZEN"
)

print(
    f"Epochs: {INITIAL_EPOCHS}"
)

print(
    f"Learning rate: {INITIAL_LEARNING_RATE}"
)

print()


history_stage1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint,
        reduce_lr
    ]
)


# ============================================================
# STAGE 2
# ============================================================

print()
print("=" * 70)
print("STAGE 2 FINE-TUNING")
print("=" * 70)


# Unfreeze backbone

base_model = model.get_layer(
    "efficientnetv2-b3"
)

base_model.trainable = True


# ------------------------------------------------------------
# Fine-tune from approximately the final 30% of layers
# ------------------------------------------------------------

total_layers = len(
    base_model.layers
)

fine_tune_from = int(
    total_layers * 0.70
)


print(
    f"Total backbone layers: {total_layers}"
)

print(
    f"Fine-tuning from layer: {fine_tune_from}"
)


for layer_index, layer in enumerate(
    base_model.layers
):

    if layer_index < fine_tune_from:

        layer.trainable = False

    else:

        layer.trainable = True


# Keep BatchNormalization frozen during fine-tuning

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# Recompile

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LEARNING_RATE
    ),
    loss=CoralLoss()
)


print()
print(
    f"Fine-tuning learning rate: "
    f"{FINE_TUNE_LEARNING_RATE}"
)

print(
    f"Stage 2 epochs: "
    f"{FINE_TUNE_EPOCHS}"
)

print()


# ============================================================
# STAGE 2 CALLBACKS
# ============================================================

stage2_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)


stage2_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# CONTINUE EPOCH NUMBERING
# ============================================================

history_stage2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    initial_epoch=INITIAL_EPOCHS,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=[
        stage2_checkpoint,
        stage2_reduce_lr
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)


model.save(
    final_model_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

import json


history = {}


for key, values in history_stage1.history.items():

    history.setdefault(
        key,
        []
    )

    history[key].extend(
        [
            float(value)
            for value in values
        ]
    )


for key, values in history_stage2.history.items():

    history.setdefault(
        key,
        []
    )

    history[key].extend(
        [
            float(value)
            for value in values
        ]
    )


history_path = os.path.join(
    REPORT_DIR,
    "clean_training_history.json"
)


with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best model : {best_model_path}"
)

print(
    f"Final model: {final_model_path}"
)

print(
    f"History    : {history_path}"
)

print("=" * 70)