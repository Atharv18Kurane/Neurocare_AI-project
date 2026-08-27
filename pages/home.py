import streamlit as st


st.title("🧠 NeuroCareAI 2.0")

st.subheader(
    "Alzheimer's MRI Analysis System"
)

st.write(
    """
    NeuroCareAI 2.0 is an AI-based system designed
    to analyze MRI images and classify Alzheimer's
    disease severity using EfficientNetV2-B3 with
    CORAL ordinal classification.
    """
)

st.divider()


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Test Accuracy",
        "96.26%"
    )


with col2:

    st.metric(
        "Model",
        "EfficientNetV2-B3"
    )


with col3:

    st.metric(
        "Classes",
        "4"
    )


st.divider()


st.subheader("Disease Classification")

st.write(
    """
    The system classifies MRI scans into four categories:
    """
)

st.markdown(
    """
    - 🟢 **NonDemented**
    - 🟡 **VeryMildDemented**
    - 🟠 **MildDemented**
    - 🔴 **ModerateDemented**
    """
)

st.info(
    "This system is intended for research and educational purposes "
    "and should not replace professional medical diagnosis."
)