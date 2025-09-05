#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line predictor for the fraud model.

Usage:
  python predict_fraud.py --model model.joblib --data fraud.csv [--output /path/to/preds.csv] [--threshold 0.5]

It replicates the cleaning and feature engineering used during training,
aligns columns to the model, and outputs predictions.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from fraud_features import clean_and_engineer, align_to_model_columns


def _infer_output_path(data_path: str) -> str:
    base, _ = os.path.splitext(data_path)
    return base + "_predictions.csv"


def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        print(f"Failed to load model from {model_path}: {exc}", file=sys.stderr)
        raise
    return model


def predict(model, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    # Use predict_proba if available, otherwise fallback to decision_function or predict
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            p1 = proba[:, 1]
        else:
            # Some estimators may output 1D proba for binary
            p1 = proba.ravel()
        y_pred = (p1 >= threshold).astype(int)
        return pd.DataFrame({
            'pred_proba': p1,
            'pred_label': y_pred,
        })
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        # Convert scores to [0,1] via logistic if possible
        p1 = 1 / (1 + np.exp(-scores))
        y_pred = (p1 >= threshold).astype(int)
        return pd.DataFrame({
            'pred_proba': p1,
            'pred_label': y_pred,
        })
    else:
        labels = model.predict(X)
        return pd.DataFrame({
            'pred_label': labels,
        })


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Fraud model predictor CLI')
    parser.add_argument('-m', '--model', required=True, help='Path to trained model artifact (e.g., joblib)')
    parser.add_argument('-d', '--data', required=True, help='Path to input CSV with raw transactions')
    parser.add_argument('-o', '--output', default=None, help='Optional output CSV path for predictions')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Classification threshold for positive class')
    args = parser.parse_args(argv)

    # Read raw data
    try:
        raw_df = pd.read_csv(args.data)
    except Exception as exc:
        print(f"Failed to read data from {args.data}: {exc}", file=sys.stderr)
        return 2

    # Apply same cleaning and feature engineering
    fe_df = clean_and_engineer(raw_df)

    # Prepare feature matrix X (drop target if present)
    X = fe_df.copy()
    if 'fraud' in X.columns:
        X = X.drop(columns=['fraud'])

    # Load model and align columns
    model = load_model(args.model)
    try:
        X_aligned = align_to_model_columns(X, model)
    except Exception as exc:
        print(f"Failed to align features to model columns: {exc}", file=sys.stderr)
        return 3

    # Predict
    preds = predict(model, X_aligned, threshold=args.threshold)

    # Attach identifiers if present for convenience
    id_cols = [c for c in ['customer', 'merchant', 'step'] if c in fe_df.columns]
    out_df = pd.concat([fe_df[id_cols].reset_index(drop=True), preds.reset_index(drop=True)], axis=1) if id_cols else preds

    # Output
    out_path = args.output or _infer_output_path(args.data)
    try:
        out_df.to_csv(out_path, index=False)
    except Exception as exc:
        print(f"Warning: failed to write predictions to {out_path}: {exc}", file=sys.stderr)

    # Also print to stdout (first few and count) to satisfy simple CLI usage
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(out_df.head(10))
    print(f"\nWrote {len(out_df)} predictions to: {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


