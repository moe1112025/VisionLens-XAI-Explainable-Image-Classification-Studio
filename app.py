from __future__ import annotations

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions as mobilenet_decode, preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions as resnet_decode, preprocess_input as resnet_preprocess

st.set_page_config(page_title="VisionLens XAI", page_icon="◉", layout="wide")
st.markdown("<style>.stApp{background:#080b10;color:#eef3f8}.block-container{max-width:1350px;padding-top:2rem}.hero h1{font-size:3rem;letter-spacing:-.05em}.hero p{color:#91a0ae}</style>", unsafe_allow_html=True)

CONFIG = {
    "MobileNetV2": {"factory": MobileNetV2, "preprocess": mobilenet_preprocess, "decode": mobilenet_decode, "layer": "Conv_1"},
    "ResNet50": {"factory": ResNet50, "preprocess": resnet_preprocess, "decode": resnet_decode, "layer": "conv5_block3_out"},
}


@st.cache_resource(show_spinner=False)
def load_model(name):
    return CONFIG[name]["factory"](weights="imagenet")


def prepare_image(image, name):
    rgb = image.convert("RGB")
    resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    return CONFIG[name]["preprocess"](np.expand_dims(array, axis=0))


def make_gradcam(data, model, layer_name, class_index):
    layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model(model.inputs, [layer.output, model.output])
    with tf.GradientTape() as tape:
        activations, predictions = grad_model(data)
        target = predictions[:, class_index]
    gradients = tape.gradient(target, activations)
    weights = tf.reduce_mean(gradients, axis=(1, 2))[0]
    activations = activations[0]
    heatmap = tf.reduce_sum(activations * weights[tf.newaxis, tf.newaxis, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    return (heatmap / (maximum + 1e-8)).numpy()


def create_overlay(heatmap, image, alpha):
    heatmap = np.clip(heatmap, 0, 1)
    rgb = np.stack([heatmap, np.sqrt(heatmap), 1.0 - heatmap], axis=-1)
    heat_image = Image.fromarray((rgb * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.asarray(heat_image, dtype=np.float32)
    result = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


st.markdown('<div class="hero"><h1>VisionLens XAI</h1><p>Image classification with visual explanation.</p></div>', unsafe_allow_html=True)
model_name = st.sidebar.selectbox("Model architecture", list(CONFIG))
show_gradcam = st.sidebar.checkbox("Enable Grad-CAM", value=True)
alpha = st.sidebar.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()
try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Invalid image: {exc}")
    st.stop()

with st.spinner(f"Loading {model_name}..."):
    model = load_model(model_name)
array = prepare_image(image, model_name)
with st.spinner("Running inference..."):
    raw = model.predict(array, verbose=0)
    predictions = CONFIG[model_name]["decode"](raw, top=5)[0]

heatmap = None
if show_gradcam:
    try:
        with st.spinner("Computing Grad-CAM..."):
            heatmap = make_gradcam(array, model, CONFIG[model_name]["layer"], int(np.argmax(raw[0])))
    except Exception as exc:
        st.warning(f"Grad-CAM unavailable: {exc}")

cols = st.columns(2 if heatmap is not None else 1)
with cols[0]:
    st.subheader("Input")
    st.image(image, use_container_width=True)
if heatmap is not None:
    with cols[1]:
        st.subheader("Grad-CAM")
        st.image(create_overlay(heatmap, image, alpha), use_container_width=True)
        st.caption("Brighter regions indicate stronger contribution to the selected class score.")

st.divider()
st.subheader("Top-5 predictions")
for _, label, score in predictions:
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"**{label.replace('_', ' ').title()}**")
    with right:
        st.progress(float(score), text=f"{float(score):.2%}")
st.caption(f"Architecture · {model_name} · ImageNet · 224×224 input")
