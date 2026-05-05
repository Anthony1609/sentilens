"""
inference.py
------------
Inference engine: loads trained artifacts and classifies text.
Designed to be imported by the API layer (app.py) or used standalone.
"""

import os
import sys
import json
import time
import logging

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

MAX_TOKENS = 512   # PRD: FR-3


class SentimentPredictor:
    """
    Wraps the trained pipeline (vectorizer + model + label encoder).
    Thread-safe for single-process WSGI/ASGI servers.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir       = model_dir
        self.model           = None
        self.vectorizer      = None
        self.label_encoder   = None
        self.metadata        = {}
        self.preprocessor    = TextPreprocessor()
        self._loaded         = False
        self.load()

    # ── Loading ───────────────────────────────────────────────────────────────
    def load(self):
        paths = {
            "model":         os.path.join(self.model_dir, "model.joblib"),
            "vectorizer":    os.path.join(self.model_dir, "vectorizer.joblib"),
            "label_encoder": os.path.join(self.model_dir, "label_encoder.joblib"),
            "metadata":      os.path.join(self.model_dir, "metadata.json"),
        }
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            logger.warning(f"[Inference] Missing artifacts: {missing}. Model not loaded.")
            self._loaded = False
            return

        self.model         = joblib.load(paths["model"])
        self.vectorizer    = joblib.load(paths["vectorizer"])
        self.label_encoder = joblib.load(paths["label_encoder"])
        with open(paths["metadata"]) as f:
            self.metadata  = json.load(f)
        self._loaded = True
        logger.info(f"[Inference] Model v{self.metadata.get('model_version','?')} loaded.")

    def is_ready(self):
        return self._loaded

    # ── Single Prediction ─────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Classify a single text string.

        Returns:
            dict with keys: sentiment_label, confidence_score,
                            probabilities, processing_time_ms,
                            model_version, truncated (bool)
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Run train.py first.")

        t0 = time.perf_counter()

        # Validate / truncate
        truncated = False
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        tokens = text.split()
        if len(tokens) > MAX_TOKENS:
            text = " ".join(tokens[:MAX_TOKENS])
            truncated = True
            logger.warning("[Inference] Input truncated to 512 tokens.")

        clean = self.preprocessor.preprocess(text)
        vec   = self.vectorizer.transform([clean])
        probs = self.model.predict_proba(vec)[0]
        idx   = int(np.argmax(probs))
        label = self.label_encoder.classes_[idx]

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        prob_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self.label_encoder.classes_, probs)
        }

        return {
            "sentiment_label":   label,
            "confidence_score":  round(float(probs[idx]), 4),
            "probabilities":     prob_dict,
            "processing_time_ms": elapsed_ms,
            "model_version":     self.metadata.get("model_version", "unknown"),
            "truncated":         truncated,
        }

    # ── Batch Prediction ──────────────────────────────────────────────────────
    def predict_batch(self, texts: list) -> dict:
        """
        Classify up to 100 texts in one call (PRD: FR-7).

        Returns:
            dict with keys: predictions (list), batch_processing_time_ms
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Run train.py first.")
        if len(texts) > 100:
            raise ValueError("Batch size cannot exceed 100 items.")

        t0 = time.perf_counter()
        results = []
        for text in texts:
            try:
                r = self.predict(text)
                results.append({
                    "sentiment_label":  r["sentiment_label"],
                    "confidence_score": r["confidence_score"],
                    "probabilities":    r["probabilities"],
                    "truncated":        r["truncated"],
                })
            except Exception as e:
                results.append({"error": str(e)})

        return {
            "predictions":            results,
            "batch_processing_time_ms": int((time.perf_counter() - t0) * 1000),
        }

    # ── Health Info ───────────────────────────────────────────────────────────
    def health(self) -> dict:
        return {
            "status":        "ready" if self._loaded else "degraded",
            "model_version": self.metadata.get("model_version", "N/A"),
            "model_type":    self.metadata.get("model_type", "N/A"),
            "classes":       self.metadata.get("classes", []),
            "macro_f1":      self.metadata.get("macro_f1", None),
        }
