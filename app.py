"""Streamlit UI.   Run with:  streamlit run app.py

Sidebar: choose a model and a photo (upload one or take a test photo).
Main area: the verdict, the probability, and a Grad-CAM heatmap showing
which part of the photo the model based its decision on."""

import json

import numpy as np
import streamlit as st
from PIL import Image

from inspector import config
from inspector.data import load_split, prepare_image
from inspector.gradcam import grad_cam, overlay
from inspector.models import load_model, model_path, predict_proba

st.set_page_config(page_title="Casting Defect Inspector", page_icon="🔍", layout="wide")

MODEL_NAMES = {"resnet18": "ResNet-18 (pretrained)",
               "resnet18_relit": "ResNet-18 + lighting augmentation",
               "small_cnn": "Small CNN (from scratch)"}
RESULTS_FILE = config.ROOT / "eval" / "results.json"


@st.cache_resource
def get_model(name: str):
    return load_model(name)


@st.cache_data
def get_test_photos():
    return load_split("test")


def saved_threshold(name: str) -> float:
    """The cost-based threshold found by eval/run_eval.py, or 0.5 before it has run."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())["test"][name]["threshold"]
    return 0.5


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("Casting Defect Inspector")
    trained = [name for name in MODEL_NAMES if model_path(name).exists()]
    if not trained:
        st.error("No trained model found. Run `python -m inspector.train resnet18` first.")
        st.stop()
    name = st.radio("Model", trained, format_func=MODEL_NAMES.get)
    threshold = st.slider("Reject the part when p(defective) is at least",
                          0.01, 0.99, saved_threshold(name), 0.01,
                          help="The default is the threshold with the lowest cost on the "
                               "validation photos.")

    st.subheader("Photo")
    upload = st.file_uploader("Upload a top-view photo of a casting", type=["jpg", "jpeg", "png"])
    images, labels = get_test_photos()
    row = st.number_input(f"Or use test photo number (0 to {len(labels) - 1})",
                          0, len(labels) - 1, 0)

# ---------------- Verdict ----------------
if upload:
    photo, truth = prepare_image(Image.open(upload)), None
else:
    photo, truth = images[row], int(labels[row])

model = get_model(name)
probability = float(predict_proba(model, photo[None])[0])
defective = probability >= threshold

left, right = st.columns(2)
left.image(photo, caption="Photo as the model sees it (224 x 224 grayscale)", width=380)
right.image(overlay(photo, grad_cam(model, photo), 0.6),
            caption="Grad-CAM: red areas pushed the model towards 'defective'", width=380)

if defective:
    st.error(f"**REJECT: defective.** p(defective) = {probability:.3f}, threshold {threshold:.2f}")
else:
    st.success(f"**PASS: ok.** p(defective) = {probability:.3f}, threshold {threshold:.2f}")

if truth is not None:
    actual = "defective" if truth else "ok"
    correct = "correct" if bool(truth) == defective else "wrong"
    st.caption(f"True label of this test photo: {actual}. The model is {correct}.")
