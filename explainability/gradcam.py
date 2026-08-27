import os
import numpy as np
import tensorflow as tf
from PIL import Image
from config.config import IMAGE_SIZE, MODEL_DIR

from models.coral_utils import (
    coral_probabilities,
    CLASS_NAMES
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "best_model.keras"
)

GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "gradcam"
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = (
    300,
    300
)


# ============================================================
# LOAD MODEL
# ============================================================

_model = None


def load_gradcam_model():

    global _model

    if _model is None:

        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    return _model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

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

    # IMPORTANT:
    # Do NOT divide by 255.
    #
    # This must exactly match models/predict.py.
    #
    # The trained EfficientNetV2-B3 model handles
    # its expected input preprocessing internally.

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# CORAL OUTPUT → CLASS PROBABILITIES
# ============================================================
def coral_to_class_probabilities(logits):

    return coral_probabilities(
        logits
    )


# ============================================================
# FIND CONNECTED EFFICIENTNET LAYER
# ============================================================

def find_backbone_layer(
    model
):

    # --------------------------------------------------------
    # Your model contains:
    #
    # efficientnetv2-b3
    #
    # We target the OUTPUT of this outer layer.
    # --------------------------------------------------------

    for layer in model.layers:

        name = layer.name.lower()

        if (
            "efficientnetv2-b3" in name
            or "efficientnetv2_b3" in name
        ):

            output = layer.output

            if len(output.shape) == 4:

                print(
                    "Grad-CAM backbone found:",
                    layer.name
                )

                print(
                    "Feature map shape:",
                    output.shape
                )

                return layer


    # --------------------------------------------------------
    # Fallback: find last 4D layer
    # --------------------------------------------------------

    for layer in reversed(
        model.layers
    ):

        try:

            output = layer.output

            if len(output.shape) == 4:

                print(
                    "Grad-CAM fallback layer:",
                    layer.name
                )

                print(
                    "Feature map shape:",
                    output.shape
                )

                return layer

        except Exception:

            continue


    raise ValueError(
        "Could not find a connected 4D feature layer "
        "for Grad-CAM."
    )


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image_array,
    class_index=None
):

    model = load_gradcam_model()

    # ========================================================
    # FIND EFFICIENTNETV2-B3 BACKBONE
    # ========================================================

    backbone = None
    backbone_index = None

    for i, layer in enumerate(model.layers):

        name = layer.name.lower()

        if (
            "efficientnetv2-b3" in name
            or "efficientnetv2_b3" in name
        ):

            backbone = layer
            backbone_index = i

            break


    if backbone is None:

        raise ValueError(
            "EfficientNetV2-B3 backbone was not found."
        )


    print(
        "Grad-CAM backbone:",
        backbone.name
    )


    # ========================================================
    # LAYERS AFTER BACKBONE
    # ========================================================

    head_layers = model.layers[
        backbone_index + 1:
    ]


    print(
        "Number of head layers:",
        len(head_layers)
    )


    # ========================================================
    # GRADIENT TAPE
    # ========================================================

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )


    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # Run EfficientNetV2-B3 directly
        # ----------------------------------------------------

        feature_maps = backbone(
            image_tensor,
            training=False
        )


        # ----------------------------------------------------
        # Watch feature maps
        # ----------------------------------------------------

        tape.watch(
            feature_maps
        )


        # ----------------------------------------------------
        # Run classifier head manually
        # ----------------------------------------------------

        x = feature_maps

        for layer in head_layers:

            x = layer(
                x,
                training=False
            )


        logits = x


        # ----------------------------------------------------
        # Convert CORAL logits to 4 class probabilities
        # ----------------------------------------------------

        probabilities = coral_to_class_probabilities(
            logits
        )


        # ----------------------------------------------------
        # Determine target class
        # ----------------------------------------------------

        if class_index is None:

            class_index = tf.argmax(
                probabilities[0]
            )


        class_score = probabilities[
            0,
            class_index
        ]


    # ========================================================
    # CALCULATE GRADIENTS
    # ========================================================

    gradients = tape.gradient(
        class_score,
        feature_maps
    )


    if gradients is None:

        raise ValueError(
            "Gradients are None. "
            "The classifier output is not connected "
            "to EfficientNetV2-B3 feature maps."
        )


    # ========================================================
    # GLOBAL AVERAGE POOLING OF GRADIENTS
    # ========================================================

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )


    # ========================================================
    # REMOVE BATCH DIMENSION
    # ========================================================

    feature_maps = feature_maps[0]

    pooled_gradients = pooled_gradients[0]


    # ========================================================
    # WEIGHT FEATURE MAPS
    # ========================================================

    heatmap = tf.reduce_sum(
        feature_maps *
        pooled_gradients,
        axis=-1
    )


    # ========================================================
    # RELU
    # ========================================================

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    max_value = tf.reduce_max(
        heatmap
    )


    heatmap = heatmap / (
        max_value + 1e-8
    )


    heatmap = heatmap.numpy()


    # ========================================================
    # RETURN
    # ========================================================

    return (
        heatmap,
        int(class_index),
        probabilities.numpy()[0]
    )
    # --------------------------------------------------------
    # IMPORTANT
    #
    # target_layer.output is the OUTER EfficientNetV2-B3
    # output and is connected to model.inputs.
    # --------------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output
        ]
    )

    # --------------------------------------------------------
    # Gradient calculation
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        feature_maps, logits = grad_model(
            image_array,
            training=False
        )

        probabilities = coral_to_class_probabilities(
            logits
        )

        if class_index is None:

            class_index = tf.argmax(
                probabilities[0]
            )

        class_score = probabilities[
            0,
            class_index
        ]

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:

        raise ValueError(
            "Gradients are None. "
            "Grad-CAM could not calculate gradients."
        )

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    feature_maps = feature_maps[0]

    pooled_gradients = pooled_gradients[0]

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        feature_maps *
        pooled_gradients,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )

    heatmap = heatmap.numpy()

    return (
        heatmap,
        int(class_index),
        probabilities.numpy()[0]
    )


# ============================================================
# CREATE HEATMAP IMAGE
# ============================================================

def create_heatmap_image(
    heatmap,
    original_size
):

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    heatmap_image = heatmap_image.resize(
        original_size
    )

    return heatmap_image


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image_path,
    heatmap,
    alpha=0.45
):

    original = Image.open(
        image_path
    ).convert("RGB")

    original_size = original.size

    heatmap_image = create_heatmap_image(
        heatmap,
        original_size
    )

    # --------------------------------------------------------
    # Use PIL to create a colored heatmap
    # --------------------------------------------------------

    heatmap_array = np.array(
        heatmap_image
    )

    # Simple red/yellow style heatmap
    colored = np.zeros(
        (
            heatmap_array.shape[0],
            heatmap_array.shape[1],
            3
        ),
        dtype=np.uint8
    )

    colored[:, :, 0] = heatmap_array

    colored[:, :, 1] = (
        heatmap_array * 0.5
    ).astype(
        np.uint8
    )

    colored[:, :, 2] = 0

    colored_image = Image.fromarray(
        colored
    )

    overlay = Image.blend(
        original,
        colored_image,
        alpha
    )

    return overlay


# ============================================================
# SAVE GRAD-CAM
# ============================================================

def save_gradcam(
    image_path
):

    image_array = preprocess_image(
        image_path
    )

    # --------------------------------------------------------
    # First prediction
    # --------------------------------------------------------

    heatmap, class_index, probabilities = generate_gradcam(
        image_array
    )

    prediction = CLASS_NAMES[
        class_index
    ]

    confidence = float(
        probabilities[
            class_index
        ]
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        image_path,
        heatmap
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    filename = os.path.basename(
        image_path
    )

    name = os.path.splitext(
        filename
    )[0]

    output_path = os.path.join(
        GRADCAM_DIR,
        f"{name}_gradcam.png"
    )

    overlay.save(
        output_path
    )

    return (
        prediction,
        confidence,
        probabilities,
        output_path
    )