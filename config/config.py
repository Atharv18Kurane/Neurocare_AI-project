import os

# ============================================================
# NeuroCareAI 2.0 Configuration
# ============================================================

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "clean"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VALIDATION_DIR = os.path.join(
    DATASET_DIR,
    "validation"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# ============================================================
# Dataset
# ============================================================

CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# EfficientNetV2-B3
# ============================================================

IMAGE_SIZE = (300, 300)
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300

CHANNELS = 3

BATCH_SIZE = 32

# ============================================================
# Training
# ============================================================

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 20

INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

RANDOM_SEED = 42

# ============================================================
# CORAL
# ============================================================

# 4 ordered classes require 3 ordinal thresholds
CORAL_OUTPUTS = NUM_CLASSES - 1

# ============================================================
# Create required directories
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)