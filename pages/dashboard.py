import streamlit as st
import pandas as pd

from database.clinical_database import (
    initialize_clinical_database,
    get_total_patients,
    get_total_predictions,
    get_today_predictions,
    get_recent_predictions
)


# ============================================================
# DATABASE
# ============================================================

initialize_clinical_database()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Doctor Dashboard")

st.write(
    "Overview of NeuroCareAI 2.0 patient and MRI analysis activity."
)


# ============================================================
# STATISTICS
# ============================================================

total_patients = get_total_patients()

total_predictions = get_total_predictions()

today_predictions = get_today_predictions()


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "👨‍⚕️ Total Patients",
        total_patients
    )


with col2:

    st.metric(
        "🧠 MRI Analyses",
        total_predictions
    )


with col3:

    st.metric(
        "📅 Today's Predictions",
        today_predictions
    )


# ============================================================
# RECENT PREDICTIONS
# ============================================================

st.divider()

st.subheader(
    "📋 Recent MRI Predictions"
)


recent_predictions = get_recent_predictions(
    10
)


if not recent_predictions:

    st.info(
        "No MRI predictions available yet."
    )

else:

    table_data = []


    for record in recent_predictions:

        (
            patient_code,
            patient_name,
            prediction,
            confidence,
            created_at
        ) = record


        table_data.append({

            "Patient ID":
                patient_code,

            "Patient":
                patient_name,

            "Prediction":
                prediction,

            "Confidence":
                f"{confidence * 100:.2f}%",

            "Date":
                created_at

        })


    dataframe = pd.DataFrame(
        table_data
    )


    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🤖 AI Model"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Architecture",
        "EfficientNetV2-B3"
    )


with col2:

    st.metric(
        "Loss",
        "CORAL"
    )


with col3:

    st.metric(
        "Test Accuracy",
        "96.26%"
    )


with col4:

    st.metric(
        "Ordinal MAE",
        "0.0383"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ NeuroCareAI 2.0 is a research/educational "
    "decision-support prototype and should not replace "
    "professional medical diagnosis."
)