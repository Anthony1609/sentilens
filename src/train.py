"""
train.py
--------
Reproducible training pipeline.

Usage:
    python train.py                        # uses built-in seed data
    python train.py --data path/to/data.csv
    python train.py --model logistic       # or 'naive_bayes'
    python train.py --model logistic --data reviews.csv --output models/
"""

import argparse
import os
import sys
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import TextPreprocessor
from dataset import load_dataset

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_VERSION = "1.0.0"
VOCAB_SIZE     = 10_000
MAX_DF         = 0.95   # ignore terms that appear in >95% of docs
MIN_DF         = 2      # ignore terms that appear in fewer than 2 docs
RANDOM_STATE   = 42


def train(data_path: str = None, model_type: str = "logistic", output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" CSC 309 — Sentiment Analysis Training Pipeline")
    print("="*60)

    df = load_dataset(data_path)
    print(f"\n[1/5] Label distribution:\n{df['label'].value_counts().to_string()}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing text...")
    t0 = time.time()
    preprocessor = TextPreprocessor()
    df["clean_text"] = preprocessor.preprocess_batch(df["text"].tolist())
    print(f"      Done in {time.time()-t0:.2f}s")

    # ── 3. Encode Labels & Split ──────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    print(f"\n[3/5] Label classes: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 4. Vectorize ──────────────────────────────────────────────────────────
    print("\n[4/5] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=VOCAB_SIZE,
        ngram_range=(1, 2),   # unigrams + bigrams for negation context
        max_df=MAX_DF,
        min_df=MIN_DF,
        sublinear_tf=True,    # log normalization
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # ── 5. Train Model ────────────────────────────────────────────────────────
    print(f"\n[5/5] Training {model_type} classifier...")
    if model_type == "logistic":
        model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    else:
        model = MultinomialNB(alpha=0.1)

    t0 = time.time()
    model.fit(X_train_vec, y_train)
    print(f"      Trained in {time.time()-t0:.2f}s")

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "─"*60)
    print(" EVALUATION RESULTS")
    print("─"*60)
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    classes = label_encoder.classes_
    header = f"{'':12}" + "".join(f"{c:12}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:12}" + "".join(f"{v:12}" for v in row))

    print(f"\n Macro F1 Score: {macro_f1:.4f}")
    status = "✓ PASS (≥ 0.80)" if macro_f1 >= 0.80 else "✗ BELOW TARGET (< 0.80)"
    print(f" Target Status : {status}")

    # ── Persist Artifacts ─────────────────────────────────────────────────────
    model_path      = os.path.join(output_dir, "model.joblib")
    vectorizer_path = os.path.join(output_dir, "vectorizer.joblib")
    encoder_path    = os.path.join(output_dir, "label_encoder.joblib")
    meta_path       = os.path.join(output_dir, "metadata.json")

    joblib.dump(model,         model_path)
    joblib.dump(vectorizer,    vectorizer_path)
    joblib.dump(label_encoder, encoder_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": model_type,
        "vocab_size": len(vectorizer.vocabulary_),
        "macro_f1": round(macro_f1, 4),
        "classes": list(label_encoder.classes_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "ngram_range": [1, 2],
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n Artifacts saved to '{output_dir}/'")
    print(f"   model.joblib | vectorizer.joblib | label_encoder.joblib | metadata.json")
    print("="*60 + "\n")

    return macro_f1, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument("--data",   default=None,      help="Path to CSV dataset")
    parser.add_argument("--model",  default="logistic", choices=["logistic","naive_bayes"])
    parser.add_argument("--output", default="models",   help="Output directory for artifacts")
    args = parser.parse_args()

    score, meta = train(args.data, args.model, args.output)
    sys.exit(0 if score >= 0.70 else 1)
