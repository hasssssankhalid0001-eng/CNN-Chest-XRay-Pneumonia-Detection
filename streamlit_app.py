import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background-color: #f4f9fb;
}

header {
    background-color: transparent !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

/* Header */

.hero {
    text-align: center;
    padding: 15px 0 35px 0;
}

.hero-icon {
    font-size: 48px;
}

.hero-title {
    font-size: 42px;
    font-weight: 750;
    color: #12304a !important;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    color: #466273 !important;
}

/* Section headings */

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #12304a !important;
    margin-bottom: 12px;
}

/* Upload */

.upload-box {
    padding: 22px 25px;
    border: 1px solid #c9e1e7;
    border-radius: 15px;
    background-color: #ffffff;
    margin-bottom: 20px;
    color: #466273 !important;
    font-size: 15px;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 2px dashed #8ccbd0;
    border-radius: 15px;
    padding: 10px;
}

[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #345466 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #345466 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #66808e !important;
}

/* Result card */

.result-card {
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 10px;
    color: #12304a !important;
}

.normal {
    background-color: #e8f6ef;
    border: 1px solid #91d2ad;
}

.pneumonia {
    background-color: #fff0f0;
    border: 1px solid #e3a1a1;
}

.result-icon {
    font-size: 40px;
}

.result-title {
    color: #12304a !important;
    font-size: 30px;
    font-weight: 750;
    margin: 5px 0 15px 0;
}

.metric-label {
    color: #55717f !important;
    font-size: 14px;
}

.metric-value {
    color: #12304a !important;
    font-size: 28px;
    font-weight: 700;
}

/* Progress bar */

[data-testid="stProgress"] {
    background-color: #dcecef;
    border-radius: 10px;
}

[data-testid="stProgress"] > div {
    background-color: #168c95;
}

/* Model information */

[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #c9e1e7;
    border-radius: 15px;
}

[data-testid="stExpander"] summary {
    color: #12304a !important;
}

[data-testid="stExpander"] p {
    color: #466273 !important;
}

/* Buttons */

.stButton > button {
    background-color: #168c95;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 650;
}

.stButton > button:hover {
    background-color: #11757c;
    color: white !important;
}

/* Captions */

.stCaption {
    color: #607985 !important;
}

/* Disclaimer */

.disclaimer {
    text-align: center;
    color: #718792 !important;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# Header

st.markdown("""
<div class="hero">

<div class="hero-icon">
🫁
</div>

<div class="hero-title">
Chest X-Ray Pneumonia Detection
</div>

<div class="hero-subtitle">
AI-powered chest X-ray analysis using MobileNetV2
</div>

</div>
""", unsafe_allow_html=True)


# Upload section

st.markdown(
    '<div class="section-title">Upload Chest X-Ray</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="upload-box">
Upload a chest X-ray image for AI-powered analysis.
<br>
Supported formats: PNG, JPG and JPEG.
</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)


if uploaded_file:

    st.write("")

    col1, col2 = st.columns([1, 1], gap="large")

    # X-ray preview

    with col1:

        st.markdown(
            '<div class="section-title">X-Ray Preview</div>',
            unsafe_allow_html=True
        )

        st.image(
            uploaded_file,
            use_container_width=True
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )

    # Analysis

    with col2:

        st.markdown(
            '<div class="section-title">Analysis Result</div>',
            unsafe_allow_html=True
        )

        analyze = st.button(
            "🔍 Analyze X-Ray",
            use_container_width=True
        )

        if analyze:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:

                with st.spinner("Analyzing X-ray..."):

                    response = requests.post(
                        API_URL,
                        files=files,
                        timeout=60
                    )

                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]
                    probability = result["probability"]
                    confidence = result["confidence"]

                    normal_probability = 1 - probability

                    # Result card

                    if prediction == "NORMAL":

                        st.markdown(
                            f"""
                            <div class="result-card normal">

                            <div class="result-icon">
                            ✓
                            </div>

                            <div class="result-title">
                            NORMAL
                            </div>

                            <div class="metric-label">
                            Model Confidence
                            </div>

                            <div class="metric-value">
                            {confidence:.2%}
                            </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="result-card pneumonia">

                            <div class="result-icon">
                            ⚠
                            </div>

                            <div class="result-title">
                            PNEUMONIA DETECTED
                            </div>

                            <div class="metric-label">
                            Model Confidence
                            </div>

                            <div class="metric-value">
                            {confidence:.2%}
                            </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.write("")

                    # Probability metrics

                    metric1, metric2 = st.columns(2)

                    with metric1:

                        st.metric(
                            "Pneumonia Probability",
                            f"{probability:.2%}"
                        )

                    with metric2:

                        st.metric(
                            "Normal Probability",
                            f"{normal_probability:.2%}"
                        )

                    st.write("")

                    # Probability bar

                    st.markdown(
                        f"""
                        <div style="
                            color:#12304a;
                            font-weight:700;
                            font-size:16px;
                            margin-bottom:6px;
                        ">
                        Pneumonia Probability
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(probability)

                else:

                    st.error(
                        "The API could not process this image."
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure Uvicorn is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. Please try again."
                )


# Model information

st.write("")

with st.expander("Model Information"):

    info1, info2, info3, info4 = st.columns(4)

    with info1:
        st.markdown("**Architecture**")
        st.write("MobileNetV2")

    with info2:
        st.markdown("**Input Size**")
        st.write("224 × 224")

    with info3:
        st.markdown("**Task**")
        st.write("Binary Classification")

    with info4:
        st.markdown("**Classes**")
        st.write("Normal / Pneumonia")


# Disclaimer

st.markdown("""
<div class="disclaimer">

⚠️ This application is an educational AI project and is not intended
to provide medical diagnosis or replace professional medical advice.

</div>
""", unsafe_allow_html=True)