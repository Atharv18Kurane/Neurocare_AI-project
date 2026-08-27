import streamlit as st

st.title("ℹ️ About NeuroCareAI")

st.subheader("NeuroCareAI 2.0")

st.write("""
NeuroCareAI 2.0 is an AI-based Alzheimer's MRI
analysis system developed using deep learning.
""")

st.write("""
The system uses EfficientNetV2-B3 as the feature
extraction backbone and CORAL ordinal classification
for Alzheimer's disease severity prediction.
""")

st.subheader("Disease Classes")

st.markdown("""
- 🟢 NonDemented
- 🟡 VeryMildDemented
- 🟠 MildDemented
- 🔴 ModerateDemented
""")

st.warning(
    "This application is intended for research and "
    "educational purposes and should not replace "
    "professional medical diagnosis."
)