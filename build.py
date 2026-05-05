"""
build.py — Run this during Render's build phase to train and persist the model.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from train import train

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(os.path.join(MODEL_DIR, "model.joblib")):
    print("[Build] Training model...")
    train(None, "logistic", MODEL_DIR)
    print("[Build] Model trained and saved.")
else:
    print("[Build] Model already exists, skipping training.")
