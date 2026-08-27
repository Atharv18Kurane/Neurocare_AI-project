import os
import tempfile
import numpy as np

import streamlit as st
from PIL import Image

from explainability.gradcam import save_gradcam
from models.coral_utils import CLASS_NAMES


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroCareAI - Grad-CAM",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔬 Grad-CAM")

st.write(
    "Explainable AI visualization for MRI predictions."
)

st.info(
    "Grad-CAM highlights the regions of the MRI image "
    "that contributed to the EfficientNetV2-B3 prediction."
)


# ============================================================
# PATIENT
# ============================================================

st.subheader("👨‍⚕️ Patient")

selected_patient_name = st.session_state.get(
    "selected_patient_name"
)

selected_patient_code = st.session_state.get(
    "selected_patient_code"
)


if selected_patient_name:

    st.success(
        f"Selected Patient: "
        f"{selected_patient_code} - "
        f"{selected_patient_name}"
    )

else:

    st.info(
        "No patient selected. "
        "You can still generate Grad-CAM for an MRI image."
    )


# ============================================================
# UPLOAD MRI
# ============================================================

st.subheader("📁 Upload MRI")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# DISPLAY ORIGINAL IMAGE
# ============================================================

image = None

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        width=400
    )


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

if uploaded_file:

    st.divider()

    generate_button = st.button(
        "🔬 Generate Grad-CAM",
        type="primary",
        use_container_width=True
    )

    if generate_button:

        temp_image_path = None

        try:

            # ------------------------------------------------
            # Create temporary image
            # ------------------------------------------------

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_image_path = temp_file.name


            # ------------------------------------------------
            # Run Grad-CAM
            # ------------------------------------------------

            with st.spinner(
                "Generating Grad-CAM..."
            ):

                (
                    prediction,
                    confidence,
                    probabilities,
                    output_path
                ) = save_gradcam(
                    temp_image_path
                )


            # ------------------------------------------------
            # Convert probabilities to NumPy
            # ------------------------------------------------

            if hasattr(
                probabilities,
                "numpy"
            ):

                probabilities = (
                    probabilities.numpy()
                )


            probabilities = np.asarray(
                probabilities,
                dtype=np.float32
            ).reshape(-1)


            # ------------------------------------------------
            # Save results in session state
            # ------------------------------------------------

            st.session_state[
                "gradcam_prediction"
            ] = prediction


            st.session_state[
                "gradcam_confidence"
            ] = confidence


            st.session_state[
                "gradcam_probabilities"
            ] = probabilities


            st.session_state[
                "gradcam_output"
            ] = output_path


            # Save original image
            st.session_state[
                "gradcam_original_image"
            ] = np.array(
                image
            )


            st.session_state[
                "gradcam_filename"
            ] = uploaded_file.name


            st.success(
                "Grad-CAM generated successfully."
            )


        except Exception as e:

            st.error(
                "Grad-CAM generation failed."
            )

            st.exception(e)


        finally:

            # ------------------------------------------------
            # Remove temporary file
            # ------------------------------------------------

            if (
                temp_image_path
                and os.path.exists(
                    temp_image_path
                )
            ):

                try:

                    os.remove(
                        temp_image_path
                    )

                except Exception:

                    pass


# ============================================================
# SHOW RESULT
# ============================================================

if (
    "gradcam_prediction"
    in st.session_state
):

    prediction = st.session_state[
        "gradcam_prediction"
    ]


    confidence = st.session_state[
        "gradcam_confidence"
    ]


    probabilities = st.session_state[
        "gradcam_probabilities"
    ]


    output_path = st.session_state[
        "gradcam_output"
    ]


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 AI Prediction"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Prediction",
            prediction
        )


    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader(
        "📊 Class Probabilities"
    )


    # Make sure probabilities are NumPy
    if hasattr(
        probabilities,
        "numpy"
    ):

        probabilities = (
            probabilities.numpy()
        )


    probability_array = np.asarray(
        probabilities,
        dtype=np.float32
    ).reshape(-1)


    # --------------------------------------------------------
    # Display each class
    # --------------------------------------------------------

    for class_name, probability in zip(
        CLASS_NAMES,
        probability_array
    ):

        st.write(
            f"**{class_name}** "
            f"— {probability * 100:.2f}%"
        )


        st.progress(
            float(probability)
        )


    # ========================================================
    # GRAD-CAM IMAGE
    # ========================================================

    st.divider()

    st.subheader(
        "🔬 Grad-CAM Visualization"
    )


    if os.path.exists(
        output_path
    ):

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # Original MRI
        # ----------------------------------------------------

        with col1:

            st.markdown(
                "### 🧠 Original MRI"
            )


            original_image_array = (
                st.session_state.get(
                    "gradcam_original_image"
                )
            )


            if original_image_array is not None:

                original_image = Image.fromarray(
                    original_image_array
                )


                st.image(
                    original_image,
                    caption="Original MRI",
                    use_container_width=True
                )


        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        with col2:

            st.markdown(
                "### 🔬 Grad-CAM"
            )


            gradcam_image = Image.open(
                output_path
            )


            st.image(
                gradcam_image,
                caption="Grad-CAM Visualization",
                use_container_width=True
            )


        st.success(
            "Grad-CAM visualization generated successfully."
        )


        st.caption(
            f"Output: {output_path}"
        )


    else:

        st.error(
            "Grad-CAM output image was not found."
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.divider()

    st.subheader(
        "ℹ️ Explanation"
    )

    st.write(
        "The Grad-CAM heatmap highlights image regions "
        "that contributed strongly to the model's prediction. "
        "Brighter regions indicate areas receiving greater "
        "attention from the neural network."
    )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ Medical Disclaimer: NeuroCareAI 2.0 is a "
    "research/educational decision-support prototype. "
    "Grad-CAM visualizations show model attention and "
    "should not be interpreted as a medical diagnosis."
)