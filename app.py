from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = Path(__file__).parent / "mnist_cnn.keras"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root { --ink: #e8f0fa; --muted: #9aaabd; --line: #263548; --accent: #57c7ff; }
      .stApp { background: radial-gradient(ellipse at 50% -20%, #173047 0, #0b111a 48%, #080c12 100%); color: var(--ink); }
      [data-testid="stHeader"] { background: transparent; }
      .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 2rem; }
      h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
      .eyebrow { color: var(--accent); font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
      .subtitle { color: var(--muted); font-size: 1rem; margin-top: -.5rem; }
      .ready-pill { display: inline-flex; gap: .5rem; align-items: center; border: 1px solid #245844; border-radius: 999px; background: #10241d; color: #8de0b3; padding: .38rem .72rem; font-size: .82rem; font-weight: 650; }
      .ready-dot { width: .48rem; height: .48rem; border-radius: 50%; background: #56d68b; box-shadow: 0 0 10px #56d68b66; }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); border-radius: 16px; background: #101823cc; }
      .panel-note { color: var(--muted); font-size: .88rem; margin-top: -.35rem; min-height: 2.3rem; }
      .stButton > button { min-height: 2.8rem; border-radius: 10px; border: 1px solid #33516c; background: #142538; color: #eaf6ff; font-weight: 650; transition: border-color .15s ease, background .15s ease; }
      .stButton > button:hover { border-color: var(--accent); background: #1b344a; color: white; }
      .stButton > button[kind="primary"] { background: #1679a8; border-color: #35a9de; color: white; }
      .stButton > button[kind="primary"]:hover { background: #218fbe; }
      [data-testid="stFileUploader"] section { border-color: #33516c; border-radius: 12px; background: #0c141e; }
      [data-testid="stProgress"] > div > div { background: #56bde8; }
      .prediction-card { min-height: 210px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid #28516a; border-radius: 16px; background: linear-gradient(145deg, #142b3a, #101823 72%); }
      .prediction-label { color: #9eb5c8; font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
      .prediction-digit { color: white; font-size: 6.2rem; font-weight: 750; line-height: 1.1; }
      .prediction-confidence { color: #8bd8fa; font-size: 1rem; }
      .footer-note { border-top: 1px solid var(--line); margin-top: 2rem; padding-top: 1rem; color: var(--muted); font-size: .82rem; }
      @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 1.5rem; }
        .prediction-digit { font-size: 4.5rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved inference model once, only when a prediction is requested."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH.name}")

    # Keep the heavy TensorFlow import out of the initial page render.
    from tensorflow import keras

    return keras.models.load_model(MODEL_PATH, compile=False)


def to_mnist_image(image: Image.Image) -> np.ndarray:
    """Convert a single digit image to a centered, normalized MNIST input."""
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype=np.uint8)

    # Use the image border to infer the background; this is more reliable when
    # a large digit occupies much of the uploaded image.
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.median(border) > 127:
        pixels = 255 - pixels

    # Remove light noise and isolate the digit.
    pixels = np.where(pixels > 35, pixels, 0).astype(np.uint8)
    ys, xs = np.nonzero(pixels)
    if xs.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    digit = Image.fromarray(pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def run_prediction(image: Image.Image, source: str) -> None:
    processed = to_mnist_image(image)
    if not np.any(processed):
        st.warning("I couldn’t find a digit in that image. Try a darker, clearer mark.")
        return

    with st.spinner("Loading the CNN and predicting…"):
        model = load_model()
        scores = model(processed[np.newaxis, ..., np.newaxis], training=False).numpy()[0]

    st.session_state["prediction"] = {
        "image": processed,
        "scores": scores,
        "source": source,
    }


model_ready = MODEL_PATH.is_file()
heading, status = st.columns([4, 1], vertical_alignment="center")
with heading:
    st.markdown('<div class="eyebrow">MNIST · CONVOLUTIONAL NEURAL NETWORK</div>', unsafe_allow_html=True)
    st.title("Handwritten Digit Recognition")
    st.markdown('<div class="subtitle">Draw a digit or upload an image to see what the CNN recognizes.</div>', unsafe_allow_html=True)
with status:
    if model_ready:
        st.markdown('<div class="ready-pill"><span class="ready-dot"></span>Model ready</div>', unsafe_allow_html=True)
    else:
        st.error("Model file missing")

if not model_ready:
    st.warning(
        "The saved model file `mnist_cnn.keras` is missing. Run `python train.py` locally, "
        "then include the resulting model file in the deployed repository."
    )

st.write("")
draw_col, upload_col = st.columns(2, gap="large")

with draw_col:
    with st.container(border=True):
        st.subheader("Draw a digit")
        st.markdown('<div class="panel-note">Use the black pen on the white canvas. The canvas toolbar includes a clear control.</div>', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=300,
            width=300,
            drawing_mode="freedraw",
            return_image_data=True,
            key="digit_canvas",
        )
        predict_drawing = st.button(
            "Predict drawing",
            type="primary",
            use_container_width=True,
            disabled=not model_ready,
            key="predict_drawing",
        )

    if predict_drawing:
        st.session_state.pop("prediction", None)
        try:
            canvas_image = getattr(canvas_result, "image_data", None)
            if canvas_image is None:
                st.warning("The drawing canvas is still loading. Please try again.")
            else:
                rgb = np.asarray(canvas_image[:, :, :3], dtype=np.uint8)
                if not np.any(rgb < 245):
                    st.warning("Draw a digit first, then select Predict drawing.")
                else:
                    run_prediction(Image.fromarray(rgb), "drawing")
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            st.error("The canvas image could not be read. Clear the canvas, draw again, and retry.")
            st.caption(f"Canvas details: {error}")
        except Exception as error:
            st.error("Prediction failed. Please try again after the model finishes loading.")
            st.caption(f"Details: {error}")

with upload_col:
    with st.container(border=True):
        st.subheader("Upload an image")
        st.markdown('<div class="panel-note">Choose a PNG or JPG image containing one clearly visible digit.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a PNG or JPG image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="digit_upload",
        )
        source_image = None
        if uploaded_file is not None:
            try:
                source_image = Image.open(uploaded_file).convert("RGB")
                st.image(source_image, caption="Uploaded image", width=220)
            except (OSError, ValueError) as error:
                st.error("That image could not be opened. Please choose another PNG or JPG file.")
                st.caption(f"Image details: {error}")

        predict_upload = st.button(
            "Predict uploaded digit",
            use_container_width=True,
            disabled=not model_ready or source_image is None,
            key="predict_upload",
        )

    if predict_upload and source_image is not None:
        st.session_state.pop("prediction", None)
        try:
            run_prediction(source_image, "upload")
        except Exception as error:
            st.error("Prediction failed. Please try again after the model finishes loading.")
            st.caption(f"Details: {error}")

prediction = st.session_state.get("prediction")
if prediction:
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    predicted_digit = int(np.argmax(scores))
    confidence = float(scores[predicted_digit])

    st.write("")
    st.divider()
    st.subheader("Prediction")
    result_col, input_col, probabilities_col = st.columns([1, 1, 2], gap="large")
    with result_col:
        st.markdown(
            f'<div class="prediction-card"><div class="prediction-label">Predicted digit</div>'
            f'<div class="prediction-digit">{predicted_digit}</div>'
            f'<div class="prediction-confidence">{confidence:.1%} confidence</div></div>',
            unsafe_allow_html=True,
        )
    with input_col:
        st.write("**Model input · 28 × 28**")
        st.image(prediction["image"], width=170, clamp=True)
        st.caption("The input is converted to grayscale, centered, and scaled to 0–1.")
    with probabilities_col:
        st.write("**Probability by digit**")
        for class_id in np.argsort(scores)[::-1]:
            probability = float(scores[class_id])
            st.progress(probability, text=f"{class_id}  ·  {probability:.1%}")

st.markdown(
    '<div class="footer-note">Dataset: MNIST (28 × 28 grayscale digits) &nbsp;·&nbsp; '
    'Model: CNN with 10 output classes &nbsp;·&nbsp; Predictions work best with one clear, centered digit.</div>',
    unsafe_allow_html=True,
)
