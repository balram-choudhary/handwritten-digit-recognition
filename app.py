import html
import math
import time
from pathlib import Path

_APP_SCRIPT_STARTED_AT = time.perf_counter()

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
      .prediction-card { min-height: 210px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid #28516a; border-radius: 16px; background: linear-gradient(145deg, #142b3a, #101823 72%); }
      .prediction-label { color: #9eb5c8; font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
      .prediction-digit { color: white; font-size: 6.2rem; font-weight: 750; line-height: 1.1; }
      .prediction-digit--number { font-size: 3.6rem; letter-spacing: .08em; }
      .prediction-confidence { color: #8bd8fa; font-size: 1rem; }
      .probability-chart { display: grid; gap: .55rem; padding: .85rem; border: 1px solid var(--line); border-radius: 12px; background: #0d151fcc; }
      .probability-row { display: grid; grid-template-columns: 1.5rem minmax(0, 1fr) 6.2rem; align-items: center; gap: .7rem; min-width: 0; padding: .28rem .38rem; border: 1px solid transparent; border-radius: 8px; }
      .probability-row--top { border-color: #28516a; background: #122533; }
      .probability-digit-label { color: #b9c8d8; font-size: .88rem; font-weight: 650; text-align: left; }
      .probability-row--top .probability-digit-label { color: #9deaff; }
      .probability-track { display: block; position: relative; width: 100%; height: .62rem; overflow: hidden; border: 1px solid #2b3b4a; border-radius: 999px; background: #1c2a37; }
      .probability-fill { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #238cb9, #51c4eb); transition: width .25s ease-out; }
      .probability-fill--top { background: linear-gradient(90deg, #31b6e8, #8beaff); box-shadow: 0 0 10px #55d7ff55; }
      .probability-value { color: #aab9c9; font-size: .8rem; font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
      .probability-row--top .probability-value { color: #a1eaff; font-weight: 700; }
      .footer-note { border-top: 1px solid var(--line); margin-top: 2rem; padding-top: 1rem; color: var(--muted); font-size: .82rem; }
      @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 1.5rem; }
        .prediction-digit { font-size: 4.5rem; }
        .prediction-digit--number { font-size: 2.8rem; }
        .probability-chart { gap: .38rem; padding: .65rem; }
        .probability-row { grid-template-columns: 1.25rem minmax(0, 1fr) 5.8rem; gap: .48rem; padding: .25rem .25rem; }
        .probability-track { height: .52rem; }
        .probability-digit-label { font-size: .82rem; }
        .probability-value { font-size: .74rem; }
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
    model_load_started_at = time.perf_counter()
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(
        f"[performance] saved_model_load_seconds="
        f"{time.perf_counter() - model_load_started_at:.3f}",
        flush=True,
    )
    return model


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


def _split_wide_digit_region(region: np.ndarray, depth: int = 0) -> list[np.ndarray]:
    """Try a conservative vertical-valley split for clearly touching digits."""
    ys, xs = np.nonzero(region)
    if xs.size == 0:
        return []
    cropped = region[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = cropped.shape
    if depth >= 8 or width <= height * 1.15:
        return [cropped]

    projection = np.count_nonzero(cropped, axis=0)
    guard = max(1, int(height * 0.22))
    if width - 2 * guard < 1:
        return [cropped]

    candidates = projection[guard:width - guard]
    valley_floor = int(candidates.min())
    if valley_floor > projection.max() * 0.22:
        return [cropped]

    valley_threshold = max(valley_floor, int(projection.max() * 0.12))
    valley_columns = np.flatnonzero(candidates <= valley_threshold) + guard
    center = (width - 1) / 2
    split_at = int(valley_columns[np.argmin(np.abs(valley_columns - center))])
    left = cropped[:, :split_at]
    right = cropped[:, split_at:]
    if left.size == 0 or right.size == 0 or not np.any(left) or not np.any(right):
        return [cropped]

    return _split_wide_digit_region(left, depth + 1) + _split_wide_digit_region(right, depth + 1)


def split_digit_regions(image: Image.Image) -> list[Image.Image]:
    """Separate a single-line number into left-to-right digit images."""
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype=np.uint8)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.median(border) > 127:
        pixels = 255 - pixels

    foreground = np.where(pixels > 35, pixels, 0).astype(np.uint8)
    ys, xs = np.nonzero(foreground)
    if xs.size == 0:
        return []

    foreground = foreground[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    active_columns = np.flatnonzero(np.any(foreground > 0, axis=0))
    runs = np.split(active_columns, np.flatnonzero(np.diff(active_columns) > 1) + 1)

    regions = []
    for run in runs:
        if run.size == 0:
            continue
        region = foreground[:, run[0]:run[-1] + 1]
        regions.extend(_split_wide_digit_region(region))

    return [Image.fromarray(region) for region in regions]


def build_probability_chart(scores: np.ndarray) -> str:
    """Render validated model probabilities as a lightweight HTML/CSS chart."""
    values = [float(score) for score in scores]
    if len(values) != 10:
        raise ValueError("Expected one probability for each digit from 0 to 9.")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Class probabilities must be finite values between 0 and 1.")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-4, abs_tol=1e-6):
        raise ValueError("Class probabilities must add up to 1.")

    ranked_digits = sorted(range(10), key=lambda digit: (-values[digit], digit))
    top_digit = ranked_digits[0]
    rows = []
    for digit in ranked_digits:
        probability = values[digit]
        digit_label = html.escape(str(digit), quote=True)
        percent_label = html.escape(f"{probability * 100.0:.9g}%", quote=True)
        bar_width = probability * 100.0
        min_width = "2px" if probability > 0.0 else "0px"
        row_class = "probability-row probability-row--top" if digit == top_digit else "probability-row"
        fill_class = "probability-fill probability-fill--top" if digit == top_digit else "probability-fill"
        rows.append(
            f'<div class="{row_class}" role="listitem" '
            f'aria-label="Digit {digit_label}: {percent_label}">'
            f'<span class="probability-digit-label">{digit_label}</span>'
            f'<span class="probability-track" aria-hidden="true">'
            f'<span class="{fill_class}" style="width: {bar_width:.9f}%; min-width: {min_width};"></span>'
            f'</span>'
            f'<span class="probability-value">{percent_label}</span>'
            f'</div>'
        )

    return (
        '<div class="probability-chart" role="list" aria-label="Prediction probabilities by digit">'
        + "".join(rows)
        + "</div>"
    )


def run_prediction(image: Image.Image, source: str) -> None:
    digit_regions = split_digit_regions(image)
    if not digit_regions:
        st.warning("I couldn’t find a digit in that image. Try a darker, clearer mark.")
        return
    if len(digit_regions) > 12:
        st.warning("Please use a single-line input with 12 digits or fewer.")
        return

    processed_digits = [to_mnist_image(region) for region in digit_regions]
    if any(not np.any(digit) for digit in processed_digits):
        st.warning("One of the digit segments was blank. Try writing with a little more space between digits.")
        return

    with st.spinner("Loading the CNN and predicting…"):
        model = load_model()
        batch = np.stack(processed_digits)[..., np.newaxis]
        scores = model(batch, training=False).numpy()

    st.session_state["prediction"] = {
        "images": processed_digits,
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
        st.subheader("Draw digits or a number")
        st.markdown('<div class="panel-note">Write digits from left to right and leave a small gap between them. Use the canvas toolbar to clear.</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="panel-note">Choose a PNG or JPG with one clear, single-line number. Leave a small gap between digits.</div>', unsafe_allow_html=True)
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
            "Predict image",
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
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]
    input_digits = prediction.get("images")
    if input_digits is None:
        input_digits = [prediction["image"]]
    predicted_digits = [int(np.argmax(digit_scores)) for digit_scores in scores]
    predicted_number = "".join(str(digit) for digit in predicted_digits)
    digit_count = len(predicted_digits)

    st.write("")
    st.divider()
    st.subheader("Prediction")
    result_col, input_col, probabilities_col = st.columns([1, 1, 2], gap="large")
    with result_col:
        result_label = "Predicted digit" if digit_count == 1 else "Predicted number"
        result_detail = (
            f"{float(scores[0, predicted_digits[0]]):.1%} confidence"
            if digit_count == 1
            else f"{digit_count} digits · confidence shown per position"
        )
        result_class = "prediction-digit" if digit_count == 1 else "prediction-digit prediction-digit--number"
        st.markdown(
            f'<div class="prediction-card"><div class="prediction-label">{html.escape(result_label)}</div>'
            f'<div class="{result_class}">{html.escape(predicted_number)}</div>'
            f'<div class="prediction-confidence">{html.escape(result_detail)}</div></div>',
            unsafe_allow_html=True,
        )
    with input_col:
        st.write("**Segmented model inputs · 28 × 28 each**" if digit_count > 1 else "**Model input · 28 × 28**")
        if digit_count == 1:
            st.image(input_digits[0], width=170, clamp=True)
        else:
            tile_size, tile_gap = 48, 4
            preview = Image.new("L", (digit_count * tile_size + (digit_count - 1) * tile_gap, tile_size), 0)
            for index, digit_image in enumerate(input_digits):
                tile = Image.fromarray((np.clip(digit_image, 0.0, 1.0) * 255).astype(np.uint8))
                tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
                preview.paste(tile, (index * (tile_size + tile_gap), 0))
            st.image(preview, width=min(420, preview.width))
            st.caption("Segments are read from left to right.")
        if digit_count == 1:
            st.caption("The input is converted to grayscale, centered, and scaled to 0–1.")
    with probabilities_col:
        if digit_count == 1:
            st.write("**Probability by digit**")
            st.markdown(build_probability_chart(scores[0]), unsafe_allow_html=True)
        else:
            st.write("**Per-position probabilities**")
            for index, (digit, digit_scores) in enumerate(zip(predicted_digits, scores)):
                confidence = float(digit_scores[digit])
                with st.expander(
                    f"Position {index + 1} · predicted {digit} · {confidence:.1%} confidence",
                    expanded=index == 0,
                ):
                    st.markdown(build_probability_chart(digit_scores), unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Dataset: MNIST (28 × 28 grayscale digits) &nbsp;·&nbsp; '
    'Model: CNN with 10 output classes &nbsp;·&nbsp; Use clear single-line writing and leave gaps between digits.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.get("_startup_timing_logged", False):
    print(
        f"[performance] app_script_startup_seconds="
        f"{time.perf_counter() - _APP_SCRIPT_STARTED_AT:.3f}",
        flush=True,
    )
    st.session_state["_startup_timing_logged"] = True
