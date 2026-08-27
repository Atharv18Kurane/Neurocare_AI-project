import streamlit as st
import pandas as pd

from database.clinical_database import (
    initialize_clinical_database,
    get_all_predictions
)


# ============================================================
# DATABASE
# ============================================================

initialize_clinical_database()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📜 Prediction History")

st.write(
    "View previous MRI analysis results."
)


# ============================================================
# GET DATA
# ============================================================

records = get_all_predictions()


if not records:

    st.info(
        "No prediction history is available yet."
    )

    st.stop()


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

data = []

for record in records:

    (
        prediction_id,
        patient_code,
        patient_name,
        image_name,
        prediction,
        confidence,
        created_at
    ) = record

    data.append({

        "ID":
            prediction_id,

        "Patient ID":
            patient_code,

        "Patient":
            patient_name,

        "MRI":
            image_name,

        "Prediction":
            prediction,

        "Confidence":
            confidence,

        "Date":
            created_at
    })


df = pd.DataFrame(data)


# ============================================================
# FILTERS
# ============================================================

st.subheader(
    "🔍 Filters"
)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Patient filter
# ------------------------------------------------------------

with col1:

    patient_list = [
        "All"
    ] + sorted(
        df["Patient ID"].unique().tolist()
    )

    selected_patient = st.selectbox(
        "Patient",
        patient_list
    )


# ------------------------------------------------------------
# Prediction filter
# ------------------------------------------------------------

with col2:

    prediction_list = [
        "All",
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented"
    ]

    selected_prediction = st.selectbox(
        "Prediction",
        prediction_list
    )


# ------------------------------------------------------------
# Confidence filter
# ------------------------------------------------------------

with col3:

    minimum_confidence = st.slider(
        "Minimum Confidence",
        min_value=0,
        max_value=100,
        value=0
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_patient != "All":

    filtered_df = filtered_df[
        filtered_df["Patient ID"]
        ==
        selected_patient
    ]


if selected_prediction != "All":

    filtered_df = filtered_df[
        filtered_df["Prediction"]
        ==
        selected_prediction
    ]


filtered_df = filtered_df[
    filtered_df["Confidence"] >=
    minimum_confidence / 100
]


# ============================================================
# RESULT COUNT
# ============================================================

st.divider()

st.write(
    f"Showing **{len(filtered_df)}** "
    f"prediction(s)"
)


# ============================================================
# DISPLAY TABLE
# ============================================================

if filtered_df.empty:

    st.warning(
        "No predictions match the selected filters."
    )

else:

    display_df = filtered_df.copy()

    display_df["Confidence"] = (
        display_df["Confidence"] * 100
    ).round(2).astype(str) + "%"


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SELECT PREDICTION
# ============================================================

st.divider()

st.subheader(
    "🔎 Prediction Details"
)


prediction_ids = filtered_df[
    "ID"
].tolist()


if prediction_ids:

    selected_id = st.selectbox(
        "Select Prediction",
        prediction_ids
    )


    selected_record = filtered_df[
        filtered_df["ID"] == selected_id
    ].iloc[0]


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### Patient"
        )

        st.write(
            f"**Patient ID:** "
            f"{selected_record['Patient ID']}"
        )

        st.write(
            f"**Patient Name:** "
            f"{selected_record['Patient']}"
        )

        st.write(
            f"**MRI:** "
            f"{selected_record['MRI']}"
        )


    with col2:

        st.markdown(
            "### AI Result"
        )

        st.write(
            f"**Prediction:** "
            f"{selected_record['Prediction']}"
        )

        st.write(
            f"**Confidence:** "
            f"{selected_record['Confidence'] * 100:.2f}%"
        )

        st.write(
            f"**Date:** "
            f"{selected_record['Date']}"
        )
# ============================================================
# OPEN GRAD-CAM
# ============================================================

st.divider()

if st.button(
    "🔬 Open Grad-CAM",
    type="primary",
    use_container_width=True
):

    st.session_state[
        "selected_patient_code"
    ] = selected_record["Patient ID"]

    st.session_state[
        "selected_patient_name"
    ] = selected_record["Patient"]

    st.session_state[
        "history_prediction"
    ] = selected_record["Prediction"]

    st.session_state[
        "history_confidence"
    ] = selected_record["Confidence"]

    st.switch_page(
        "pages/gradcam.py"
    )