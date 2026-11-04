from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

MODEL_PATH = Path(__file__).parent / "mnist_cnn.keras"

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="wide")
st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit or upload an image. A convolutional neural network trained on MNIST will predict the digit.")


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return keras.models.load_model(MODEL_PATH)
    return None


def train_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0
    model = build_model()
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    get_model.clear()
    return float(accuracy)


def to_mnist_image(image: Image.Image) -> np.ndarray:
    """Convert an image to a centered 28x28 grayscale MNIST-style input."""
    image = ImageOps.grayscale(image)
    pixels = np.asarray(image, dtype=np.uint8)

    # Turn light-background images into white ink on black, as in MNIST.
    if pixels.mean() > 127:
        pixels = 255 - pixels

    # Discard faint noise and find the handwritten mark.
    pixels = np.where(pixels > 35, pixels, 0).astype(np.uint8)
    ys, xs = np.nonzero(pixels)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    crop = Image.fromarray(pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


with st.sidebar:
    st.header("Model")
    st.caption("Dataset: MNIST · Model: CNN · Classes: digits 0–9")
    if MODEL_PATH.exists():
        st.success("Trained model is ready")
    else:
        st.info("The model trains the first time you click the button.")
    if st.button("Train / retrain model", use_container_width=True):
        with st.spinner("Downloading MNIST and training the CNN…"):
            accuracy = train_model()
        st.success(f"Training complete · test accuracy: {accuracy:.1%}")
        st.rerun()

model = get_model()
if model is None:
    st.info("Use **Train / retrain model** in the sidebar to get started. MNIST will download automatically on the first run.")
    st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Draw a digit")
    st.caption("Use the black brush on the white canvas. Clear it to start again.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=22,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key="digit_canvas",
    )

    if st.button("Predict drawing", type="primary", use_container_width=True):
        canvas_image = (
            canvas_result.image_data
            if canvas_result is not None
            else None
        )

        if canvas_image is None:
            st.warning("Please draw a digit first.")
        else:
            rgb_image = canvas_image[:, :, :3].astype("uint8")

            if not np.any(rgb_image < 245):
                st.warning("Draw a digit first.")
            else:
                drawn = Image.fromarray(rgb_image, mode="RGB")
                st.session_state["digit_input"] = to_mnist_image(drawn)
                
with right:
    st.subheader("Or upload an image")
    st.caption("Use a clear image containing one digit. JPG and PNG are supported.")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded is not None:
        source = Image.open(uploaded).convert("RGB")
        st.image(source, caption="Uploaded image", width=180)
        if st.button("Predict uploaded digit", use_container_width=True):
            st.session_state["digit_input"] = to_mnist_image(source)

if "digit_input" in st.session_state:
    digit = st.session_state["digit_input"]
    probabilities = model.predict(digit.reshape(1, 28, 28, 1), verbose=0)[0]
    predicted = int(np.argmax(probabilities))
    st.divider()
    result, chart = st.columns([1, 2])
    with result:
        st.metric("Prediction", predicted)
        st.write(f"Confidence: **{probabilities[predicted]:.1%}**")
        st.image(digit, caption="Preprocessed 28 × 28 input", width=140, clamp=True)
    with chart:
        st.subheader("Class probabilities")
        st.bar_chart({"Probability": probabilities}, height=260)

st.caption("MNIST contains 28 × 28 grayscale images of handwritten digits. Predictions can be less reliable for handwriting unlike the training examples.")
