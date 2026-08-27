import streamlit as st

from auth.authentication import (
    initialize_auth_database,
    authenticate_doctor
)


def show_login():

    initialize_auth_database()

    st.markdown(
        "<h1 style='text-align:center;'>🧠</h1>",
        unsafe_allow_html=True
    )

    st.title("NeuroCareAI")

    st.markdown(
        "<p style='text-align:center; font-size:20px;'>"
        "Doctor Login"
        "</p>",
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("🔐 Doctor Authentication")

    email = st.text_input(
        "Doctor Email",
        placeholder="doctor@neurocare.ai"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password"
    )

    login_button = st.button(
        "🔐 Login",
        use_container_width=True
    )

    if login_button:

        if not email or not password:

            st.warning(
                "Please enter both email and password."
            )

        else:

            doctor = authenticate_doctor(
                email,
                password
            )

            if doctor:

                st.session_state.authenticated = True
                st.session_state.doctor_name = doctor[1]

                st.success(
                    f"Welcome, Dr. {doctor[1]}!"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid email or password."
                )

    st.divider()

    st.caption(
        "🔒 Authorized medical personnel only."
    )

    st.caption(
        "NeuroCareAI 2.0 | Alzheimer MRI Analysis System"
    )