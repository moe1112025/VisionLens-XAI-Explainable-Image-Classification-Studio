# VisionLens XAI — Explainable Image Classification Studio

VisionLens XAI is a standalone image-classification studio that compares pretrained MobileNetV2 and ResNet50 ImageNet classifiers and produces Grad-CAM visual explanations.

## Capabilities

- MobileNetV2 inference
- ResNet50 inference
- Top-5 ImageNet predictions
- Grad-CAM visualization
- Adjustable explanation overlay
- Cached model loading
- Image validation and controlled preprocessing
- Professional Streamlit UI

## Run

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

The pretrained weights download automatically on first use when they are not already cached.

## Architecture

```text
Image -> RGB -> 224x224 preprocessing -> CNN -> Top-5 predictions
                                      -> Grad-CAM -> explanation overlay
```

Grad-CAM is an interpretability aid rather than proof of causal reasoning. The classifier is limited to ImageNet categories.

## Technology

TensorFlow: https://www.tensorflow.org/
Keras Applications: https://keras.io/api/applications/
Streamlit: https://docs.streamlit.io/
Pillow: https://pillow.readthedocs.io/

## Windows Note

TensorFlow 2.21.0 publishes a CPython 3.11 Windows x86-64 wheel, matching the Python 3.11 environment used for the other refreshed projects.

## License

Project source: MIT. TensorFlow, pretrained model weights and upstream ImageNet-related terms remain applicable.
