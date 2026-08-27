import os
import sys
import tempfile
from datetime import datetime
import streamlit as st
from PIL import Image


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORTS
# ============================================================

from models.predict import predict_image

from explainability.gradcam import (
    save_gradcam
)

from database.clinical_database import (
    initialize_clinical_database,
    get_patients,
    save_prediction
)


# ============================================================
# DATABASE
# ============================================================

initialize_clinical_database()


# ============================================================
# STORAGE DIRECTORIES
# ============================================================

MRI_STORAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "patient_mri"
)

GRADCAM_STORAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "gradcam"
)


os.makedirs(
    MRI_STORAGE_DIR,
    exist_ok=True
)

os.makedirs(
    GRADCAM_STORAGE_DIR,
    exist_ok=True
)


# ============================================================
# PAGE
# ============================================================

st.title("🧠 MRI Analysis")

st.write(
    "Select a patient and upload an MRI scan "
    "for Alzheimer's disease analysis."
)


# ============================================================
# PATIENTS
# ============================================================

patients = get_patients()


if not patients:

    st.warning(
        "No patients registered."
    )

    st.info(
        "Please go to the Patients page and "
        "register a patient first."
    )

    st.stop()


# ============================================================
# PATIENT SELECTION
# ============================================================

patient_options = {
    f"{patient[1]} - {patient[2]}":
    patient[0]
    for patient in patients
}


selected_patient_id = st.session_state.get(
    "selected_patient_id"
)


if selected_patient_id is not None:

    selected_patient = next(
        (
            patient
            for patient in patients
            if patient[0] == selected_patient_id
        ),
        None
    )


    if selected_patient is not None:

        patient_id = selected_patient[0]

        patient_code = selected_patient[1]

        patient_name = selected_patient[2]


        st.success(
            f"👨‍⚕️ Selected Patient: "
            f"{patient_code} - {patient_name}"
        )


    else:

        st.error(
            "Selected patient was not found."
        )

        st.stop()


else:

    selected_patient = st.selectbox(
        "👨‍⚕️ Select Patient",
        list(patient_options.keys())
    )


    patient_id = patient_options[
        selected_patient
    ]


# ============================================================
# CHANGE PATIENT
# ============================================================

if st.session_state.get(
    "selected_patient_id"
) is not None:

    if st.button(
        "🔄 Change Patient"
    ):

        st.session_state.pop(
            "selected_patient_id",
            None
        )

        st.session_state.pop(
            "selected_patient_code",
            None
        )

        st.session_state.pop(
            "selected_patient_name",
            None
        )

        st.rerun()


# ============================================================
# MRI UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "🧠 Upload MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a brain MRI image."
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.divider()

    col1, col2 = st.columns(
        [1, 1]
    )


    # ========================================================
    # ORIGINAL MRI
    # ========================================================

    with col1:

        st.subheader(
            "Original MRI"
        )

        preview_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            preview_image,
            caption=uploaded_file.name,
            use_container_width=True
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    with col2:

        st.subheader(
            "AI Analysis"
        )

        analyze_button = st.button(
            "🔍 Analyze MRI",
            use_container_width=True
        )


        if analyze_button:

            # ------------------------------------------------
            # CREATE UNIQUE FILE NAME
            # ------------------------------------------------

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )


            extension = os.path.splitext(
                uploaded_file.name
            )[1]


            safe_filename = (
                f"{patient_id}_"
                f"{timestamp}"
                f"{extension}"
            )


            permanent_mri_path = os.path.join(
                MRI_STORAGE_DIR,
                safe_filename
            )


            # ------------------------------------------------
            # SAVE MRI PERMANENTLY
            # ------------------------------------------------

            with open(
                permanent_mri_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            try:

                # ====================================================
                # AI PREDICTION
                # ====================================================

                with st.spinner(
                    "Analyzing MRI using EfficientNetV2-B3..."
                ):

                    (
                        prediction,
                        confidence,
                        probabilities
                    ) = predict_image(
                        permanent_mri_path
                    )


                # ====================================================
                # GENERATE GRAD-CAM
                # ====================================================

                with st.spinner(
                    "Generating Grad-CAM explanation..."
                ):

                    (
                        gradcam_prediction,
                        gradcam_confidence,
                        gradcam_probabilities,
                        generated_gradcam_path
                    ) = save_gradcam(
                        permanent_mri_path
                    )


                # ====================================================
                # COPY GRAD-CAM INTO DATA/GRADCAM
                # ====================================================

                gradcam_filename = os.path.basename(
                    generated_gradcam_path
                )


                permanent_gradcam_path = os.path.join(
                    GRADCAM_STORAGE_DIR,
                    gradcam_filename
                )


                if (
                    os.path.abspath(
                        generated_gradcam_path
                    )
                    !=
                    os.path.abspath(
                        permanent_gradcam_path
                    )
                ):

                    import shutil

                    shutil.copy2(
                        generated_gradcam_path,
                        permanent_gradcam_path
                    )


                # ====================================================
                # SAVE PREDICTION IN DATABASE
                # ====================================================

                prediction_id = save_prediction(

                    patient_id=patient_id,

                    doctor_name=st.session_state.get(
                        "doctor_name",
                        "Unknown"
                    ),

                    image_name=uploaded_file.name,

                    prediction=prediction,

                    confidence=confidence,

                    probabilities=probabilities,

                    image_path=permanent_mri_path,

                    gradcam_path=permanent_gradcam_path

                )


                # ====================================================
                # SAVE SESSION RESULT
                # ====================================================

                st.session_state[
                    "last_prediction"
                ] = prediction


                st.session_state[
                    "last_confidence"
                ] = confidence


                st.session_state[
                    "last_probabilities"
                ] = probabilities


                st.session_state[
                    "last_filename"
                ] = uploaded_file.name


                st.session_state[
                    "last_patient_id"
                ] = patient_id


                st.session_state[
                    "last_image_path"
                ] = permanent_mri_path


                st.session_state[
                    "last_gradcam_path"
                ] = permanent_gradcam_path


                st.session_state[
                    "last_prediction_id"
                ] = prediction_id


                st.session_state[
                    "analysis_completed"
                ] = True


                st.success(
                    "MRI analysis, Grad-CAM and database "
                    "saving completed successfully."
                )


            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

                st.exception(error)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if st.session_state.get(
    "analysis_completed",
    False
):

    st.divider()

    st.subheader(
        "📋 Prediction Result"
    )


    prediction = st.session_state[
        "last_prediction"
    ]


    confidence = st.session_state[
        "last_confidence"
    ]


    probabilities = st.session_state[
        "last_probabilities"
    ]


    # ========================================================
    # RESULT CARDS
    # ========================================================

    col1, col2 = st.columns(
        2
    )


    with col1:

        st.markdown(
            "### Predicted Category"
        )


        if prediction == "NonDemented":

            st.success(
                f"🟢 {prediction}"
            )

        elif prediction == "VeryMildDemented":

            st.warning(
                f"🟡 {prediction}"
            )

        elif prediction == "MildDemented":

            st.warning(
                f"🟠 {prediction}"
            )

        else:

            st.error(
                f"🔴 {prediction}"
            )


    with col2:

        st.markdown(
            "### Confidence"
        )

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader(
        "📊 Class Probabilities"
    )


    for class_name, probability in probabilities.items():

        st.write(
            f"**{class_name}** — "
            f"{probability * 100:.2f}%"
        )


        st.progress(
            min(
                max(
                    float(probability),
                    0.0
                ),
                1.0
            )
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.divider()

    st.subheader(
        "🔬 Explainable AI"
    )


    st.write(
        "Grad-CAM highlights regions that contributed "
        "to the EfficientNetV2-B3 prediction."
    )


    gradcam_path = st.session_state.get(
        "last_gradcam_path"
    )


    if (
        gradcam_path
        and
        os.path.exists(
            gradcam_path
        )
    ):

        col1, col2 = st.columns(
            2
        )


        with col1:

            st.image(
                st.session_state[
                    "last_image_path"
                ],
                caption="Original MRI",
                use_container_width=True
            )


        with col2:

            st.image(
                gradcam_path,
                caption="Grad-CAM",
                use_container_width=True
            )


        st.success(
            "Grad-CAM generated and saved."
        )


    else:

        st.warning(
            "Grad-CAM image is not available."
        )


    # ========================================================
    # DATABASE INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "💾 Saved Record"
    )


    st.write(
        f"**Prediction ID:** "
        f"{st.session_state.get('last_prediction_id')}"
    )


    st.write(
        f"**MRI:** "
        f"{st.session_state.get('last_image_path')}"
    )


    st.write(
        f"**Grad-CAM:** "
        f"{st.session_state.get('last_gradcam_path')}"
    )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ NeuroCareAI 2.0 is a research/educational "
    "decision-support prototype and should not replace "
    "professional medical diagnosis."
)