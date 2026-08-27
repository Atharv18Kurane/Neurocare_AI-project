import streamlit as st
import pandas as pd

from database.clinical_database import (
    initialize_clinical_database,
    add_patient,
    get_patients,
    get_patient_statistics,
    get_patient_predictions
)


# ============================================================
# DATABASE
# ============================================================

initialize_clinical_database()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("👨‍⚕️ Patients")

st.write(
    "Manage registered patients and review their MRI analysis."
)


# ============================================================
# TOP CONTROLS
# ============================================================

col1, col2 = st.columns(
    [4, 1]
)

with col1:

    search = st.text_input(
        "🔍 Search Patient",
        placeholder="Search by patient ID or name",
        label_visibility="collapsed"
    )


with col2:

    add_patient_button = st.button(
        "➕ Add Patient",
        use_container_width=True
    )


# ============================================================
# ADD PATIENT FORM
# ============================================================

if add_patient_button:

    st.session_state[
        "show_add_patient"
    ] = True


if st.session_state.get(
    "show_add_patient",
    False
):

    with st.container(
        border=True
    ):

        st.subheader(
            "➕ Register New Patient"
        )

        col1, col2 = st.columns(2)

        with col1:

            patient_code = st.text_input(
                "Patient ID",
                placeholder="P001"
            )

            patient_name = st.text_input(
                "Patient Name"
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=60
            )

        with col2:

            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female",
                    "Other",
                    "Prefer not to say"
                ]
            )

            phone = st.text_input(
                "Phone"
            )


        save_col, cancel_col = st.columns(2)


        with save_col:

            save_button = st.button(
                "💾 Save Patient",
                use_container_width=True
            )


        with cancel_col:

            cancel_button = st.button(
                "Cancel",
                use_container_width=True
            )


        if cancel_button:

            st.session_state[
                "show_add_patient"
            ] = False

            st.rerun()


        if save_button:

            if not patient_code.strip():

                st.error(
                    "Patient ID is required."
                )

            elif not patient_name.strip():

                st.error(
                    "Patient name is required."
                )

            else:

                success, patient_id, message = add_patient(

                    patient_code.strip(),

                    patient_name.strip(),

                    age,

                    gender,

                    phone.strip()

                )

                if success:

                    st.success(
                        message
                    )

                    st.session_state[
                        "show_add_patient"
                    ] = False

                    st.rerun()

                else:

                    st.error(
                        message
                    )


# ============================================================
# GET PATIENTS
# ============================================================

patients = get_patients()


# ============================================================
# SEARCH
# ============================================================

if search:

    search_lower = search.lower()

    patients = [

        patient

        for patient in patients

        if (
            search_lower in patient[1].lower()
            or
            search_lower in patient[2].lower()
        )

    ]


# ============================================================
# PATIENT COUNT
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Patients",
        len(get_patients())
    )

with col2:

    st.metric(
        "Showing",
        len(patients)
    )

with col3:

    st.metric(
        "System",
        "Active"
    )


# ============================================================
# PATIENT LIST
# ============================================================

st.subheader(
    "📋 Patient Records"
)


if not patients:

    st.info(
        "No patients found."
    )

else:

    for patient in patients:

        (
            patient_id,
            patient_code,
            name,
            age,
            gender,
            phone,
            created_at
        ) = patient


        statistics = get_patient_statistics(
            patient_id
        )


        total_predictions = statistics[0] or 0

        last_date = statistics[1] or "No analysis"

        last_prediction = statistics[2] or "No prediction"

        last_confidence = statistics[3]


        # ====================================================
        # PATIENT CARD
        # ====================================================

        with st.container(
            border=True
        ):

            col1, col2, col3, col4, col5 = st.columns(
                [1.2, 2.2, 0.8, 1.2, 1.8]
            )


            with col1:

                st.markdown(
                    f"**{patient_code}**"
                )


            with col2:

                st.markdown(
                    f"**{name}**"
                )

                st.caption(
                    f"Registered: {created_at}"
                )


            with col3:

                st.write(
                    f"Age {age}"
                )


            with col4:

                st.write(
                    gender
                )


            with col5:

                if last_prediction == "NonDemented":

                    st.success(
                        "🟢 NonDemented"
                    )

                elif last_prediction == "VeryMildDemented":

                    st.warning(
                        "🟡 Very Mild"
                    )

                elif last_prediction == "MildDemented":

                    st.warning(
                        "🟠 Mild"
                    )

                elif last_prediction == "ModerateDemented":

                    st.error(
                        "🔴 Moderate"
                    )

                else:

                    st.info(
                        "No analysis"
                    )


            # =================================================
            # DETAILS
            # =================================================

            with st.expander(
                "View Patient Details"
            ):

                detail1, detail2, detail3 = st.columns(3)


                with detail1:

                    st.write(
                        f"**Patient ID:** {patient_code}"
                    )

                    st.write(
                        f"**Name:** {name}"
                    )

                    st.write(
                        f"**Age:** {age}"
                    )


                with detail2:

                    st.write(
                        f"**Gender:** {gender}"
                    )

                    st.write(
                        f"**Phone:** {phone or 'Not provided'}"
                    )


                with detail3:

                    st.write(
                        f"**Total MRI Analyses:** "
                        f"{total_predictions}"
                    )

                    st.write(
                        f"**Last Analysis:** "
                        f"{last_date}"
                    )

                    if last_confidence is not None:

                        st.write(
                            f"**Last Confidence:** "
                            f"{last_confidence * 100:.2f}%"
                        )


                # =============================================
                # PATIENT HISTORY
                # =============================================

                st.divider()

                st.markdown(
                    "### 📜 Patient MRI History"
                )


                patient_history = get_patient_predictions(
                    patient_id
                )


                if not patient_history:

                    st.info(
                        "No MRI analysis has been performed."
                    )

                else:

                    history_data = []

                    for record in patient_history:

                        (
                            prediction_id,
                            image_name,
                            prediction,
                            confidence,
                            date
                        ) = record


                        history_data.append({

                            "ID":
                                prediction_id,

                            "MRI":
                                image_name,

                            "Prediction":
                                prediction,

                            "Confidence":
                                f"{confidence * 100:.2f}%",

                            "Date":
                                date

                        })


                    dataframe = pd.DataFrame(
                        history_data
                    )


                    st.dataframe(
                        dataframe,
                        use_container_width=True,
                        hide_index=True
                    )
                st.divider()
                st.divider()

if st.button(
    "🧠 Analyze MRI",
    key=f"analyze_{patient_id}",
    use_container_width=True
):
    st.session_state["selected_patient_id"] = patient_id
    st.session_state["selected_patient_code"] = patient_code
    st.session_state["selected_patient_name"] = name

    st.switch_page(
        "pages/predict.py"
    )

