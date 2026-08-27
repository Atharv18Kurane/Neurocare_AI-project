import streamlit as st

st.title("📊 Model Performance")

st.subheader("EfficientNetV2-B3 + CORAL")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "96.26%")

with col2:
    st.metric("Precision", "96.49%")

with col3:
    st.metric("Recall", "96.50%")

with col4:
    st.metric("F1 Score", "96.49%")

st.divider()

st.subheader("Model Information")

st.write("""
**Model:** EfficientNetV2-B3

**Loss Function:** CORAL Ordinal Loss

**Classes:** 4

**Test Images:** 6,601

**Ordinal MAE:** 0.0383
""")

st.info(
    "Confusion matrix and training graphs will be displayed here."
)