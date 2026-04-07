
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json, numpy as np

@st.cache_resource
def load_model():
    with open("model_config.json") as f:
        cfg = json.load(f)
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(1280, 256),
        nn.BatchNorm1d(256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load("fake_detector_weights.pth",
                                      map_location="cpu"))
    model.eval()
    return model, cfg

def predict(img, model, cfg):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"]),
    ])
    tensor = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    label = "FAKE" if prob >= cfg["threshold"] else "REAL"
    return prob, label

st.title("Fake Image Detector")
st.write("Upload an image to check if it is AI-generated.")

model, cfg = load_model()
uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded:
    img  = Image.open(uploaded)
    prob, label = predict(img, model, cfg)
    col1, col2 = st.columns(2)
    col1.image(img, caption="Uploaded image", use_column_width=True)
    color = "red" if label == "FAKE" else "green"
    col2.markdown(f"### Prediction: :{color}[{label}]")
    col2.metric("Fake probability", f"{prob:.4f}")
    col2.metric("Threshold", f'{cfg["threshold"]}')
    col2.progress(float(prob))
    if label == "FAKE":
        col2.warning("This image appears to be AI-generated.")
    else:
        col2.success("This image appears to be a real photograph.")
