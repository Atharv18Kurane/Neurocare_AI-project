import streamlit as st

from login import show_login


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroCareAI 2.0",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "doctor_name" not in st.session_state:
    st.session_state.doctor_name = ""


# ============================================================
# LOGIN PAGE
# ============================================================

login_page = st.Page(
    show_login,
    title="Doctor Login",
    icon="🔐"
)


# ============================================================
# APPLICATION PAGES
# ============================================================

home_page = st.Page(
    "pages/home.py",
    title="Home",
    icon="🏠",
    default=True
)

patients_page = st.Page(
    "pages/patients.py",
    title="Patients",
    icon="👨‍⚕️"
)

dashboard_page = st.Page(
    "pages/dashboard.py",
    title="Dashboard",
    icon="📊"
)

predict_page = st.Page(
    "pages/predict.py",
    title="Predict",
    icon="🧠"
)

gradcam_page = st.Page(
    "pages/gradcam.py",
    title="GradCAM",
    icon="🔬"
)

performance_page = st.Page(
    "pages/performance.py",
    title="Performance",
    icon="📊"
)


history_page = st.Page(
    "pages/history.py",
    title="History",
    icon="📜"
)
about_page = st.Page(
    "pages/about.py",
    title="About",
    icon="ℹ️"
)

# ============================================================
# NOT LOGGED IN
# ============================================================

if not st.session_state.authenticated:

    pg = st.navigation(
        [login_page],
        position="hidden"
    )

    pg.run()

    st.stop()


# ============================================================
# LOGGED IN NAVIGATION
# ============================================================

pg = st.navigation(
    [
        home_page,
        patients_page,
        dashboard_page,
        predict_page,
        gradcam_page,
        performance_page,
        history_page,
        about_page
        
    ],
    position="sidebar"
)


# ============================================================
# SIDEBAR DOCTOR INFORMATION
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown(
        f"### 👨‍⚕️ Dr. {st.session_state.doctor_name}"
    )

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.authenticated = False
        st.session_state.doctor_name = ""

        st.rerun()


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()