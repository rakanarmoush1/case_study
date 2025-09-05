#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature engineering and cleaning utilities for the fraud dataset.

Replicates the transformations performed in fraud.py for inference-time use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, List


_CAT_COLS = ['customer', 'age', 'gender', 'zipcodeOri', 'merchant', 'zipMerchant', 'category']


def _strip_and_stringify(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype('string').str.strip().str.strip("'").str.strip('"')
    return df


def _make_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encode selected categoricals
    if 'age' in df.columns:
        df['age'] = df['age'].replace('U', 'ent')
        age_dummies = pd.get_dummies(df['age'], prefix='age', dtype='uint8')
        df = pd.concat([df.drop(columns=['age']), age_dummies], axis=1)

    if 'gender' in df.columns:
        gender_dummies = pd.get_dummies(df['gender'], prefix='gender', dtype='uint8')
        df = pd.concat([df.drop(columns=['gender']), gender_dummies], axis=1)

    if 'category' in df.columns:
        category_dummies = pd.get_dummies(df['category'], prefix='category', dtype='uint8')
        df = pd.concat([df.drop(columns=['category']), category_dummies], axis=1)

    if 'merchant' in df.columns:
        merchant_dummies = pd.get_dummies(df['merchant'], prefix='merchant', dtype='uint8')
        # Keep original merchant column for later stats merge; append dummies
        df = pd.concat([df, merchant_dummies], axis=1)

    return df


def _add_time_and_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Time-based features
    if 'step' in df.columns:
        df['hour_of_day'] = df['step'] % 24
        df['day_of_week'] = (df['step'] // 24) % 7

    # Amount transform
    if 'amount' in df.columns:
        df['amount'] = df['amount'].fillna(0)
        df['amount'] = np.log1p(df['amount'])

    # Ensure sort for sequential features
    if 'customer' in df.columns and 'step' in df.columns:
        df = df.sort_values(['customer', 'step']).reset_index(drop=True)

        # Cumulative/expanding stats per customer
        df['cust_txn_count'] = df.groupby('customer').cumcount()

        if 'amount' in df.columns:
            df['cust_avg_spend'] = (
                df.groupby('customer')['amount']
                .transform(lambda x: x.shift().expanding().mean())
            )
            df['cust_std_spend'] = (
                df.groupby('customer')['amount']
                .transform(lambda x: x.shift().expanding().std())
            )
        else:
            df['cust_avg_spend'] = 0.0
            df['cust_std_spend'] = 0.0

        if 'fraud' in df.columns:
            df['cust_prev_fraud'] = (
                df.groupby('customer')['fraud']
                .transform(lambda x: x.shift().fillna(0).cumsum().clip(upper=1))
            )
        else:
            df['cust_prev_fraud'] = 0

        df['cust_avg_spend'] = df['cust_avg_spend'].fillna(0)
        df['cust_std_spend'] = df['cust_std_spend'].fillna(0)
        df['cust_prev_fraud'] = df['cust_prev_fraud'].fillna(0).astype(int)

        # Rolling 24h features using two-pointer within each customer
        if 'amount' in df.columns:
            def _rolling_24h_features(group: pd.DataFrame) -> pd.DataFrame:
                steps = group['step'].values
                amounts = group['amount'].values
                txn_count_24h: List[int] = []
                amount_sum_24h: List[float] = []
                left = 0
                for right in range(len(steps)):
                    while steps[right] - steps[left] > 23:
                        left += 1
                    txn_count_24h.append(right - left)
                    amount_sum_24h.append(amounts[left:right].sum() if right > left else 0.0)
                group['txn_count_24h'] = txn_count_24h
                group['amount_sum_24h'] = amount_sum_24h
                return group

            df = df.groupby('customer', group_keys=False).apply(_rolling_24h_features)
        else:
            df['txn_count_24h'] = 0
            df['amount_sum_24h'] = 0.0
    else:
        # Fallback when cannot sort by customer/step
        df['cust_txn_count'] = 0
        df['cust_avg_spend'] = 0.0
        df['cust_std_spend'] = 0.0
        df['cust_prev_fraud'] = 0
        df['txn_count_24h'] = 0
        df['amount_sum_24h'] = 0.0

    return df


def _add_merchant_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Compute per-merchant stats. If labels not present, use counts only and default fraud rate to 0.
    if 'merchant' not in df.columns:
        df['merchant_txn_count'] = 0.0
        df['merchant_fraud_rate'] = 0.0
        return df

    if 'fraud' in df.columns:
        tmp = df[['merchant']].copy()
        tmp['fraud'] = df['fraud'].values
        agg = tmp.groupby('merchant').agg(
            merchant_txn_count=('fraud', 'count'),
            merchant_fraud_rate=('fraud', 'mean'),
        ).reset_index()
        df = df.merge(agg, on='merchant', how='left')
        global_txn_count = agg['merchant_txn_count'].mean()
        global_fraud_rate = agg['merchant_fraud_rate'].mean()
        df['merchant_txn_count'] = df['merchant_txn_count'].fillna(global_txn_count)
        df['merchant_fraud_rate'] = df['merchant_fraud_rate'].fillna(global_fraud_rate)
    else:
        # No labels available; approximate with counts in current data and 0 fraud rate
        cnt = df.groupby('merchant').size().rename('merchant_txn_count').reset_index()
        df = df.merge(cnt, on='merchant', how='left')
        df['merchant_txn_count'] = df['merchant_txn_count'].fillna(0)
        df['merchant_fraud_rate'] = 0.0

    return df


def clean_and_engineer(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and feature engineering to match training-time logic."""
    df = raw_df.copy()

    # Basic cleaning for categoricals
    df = _strip_and_stringify(df, _CAT_COLS)

    # Drop unused zip codes
    to_drop = [c for c in ['zipcodeOri', 'zipMerchant'] if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # One-hot encoding for selected columns
    df = _make_dummies(df)

    # Time and customer features
    df = _add_time_and_customer_features(df)

    # Merchant stats
    df = _add_merchant_stats(df)

    return df


def align_to_model_columns(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align runtime features to the model's expected feature columns and order."""
    expected_cols: List[str] | None = None

    # Try LightGBM attributes first
    if hasattr(model, 'feature_name_') and getattr(model, 'feature_name_') is not None:
        expected_cols = list(getattr(model, 'feature_name_'))
    elif hasattr(model, 'booster_') and model.booster_ is not None:
        try:
            expected_cols = list(model.booster_.feature_name())
        except Exception:
            expected_cols = None

    # Try scikit-learn style
    if expected_cols is None and hasattr(model, 'feature_names_in_'):
        expected_cols = list(getattr(model, 'feature_names_in_'))

    # If still unknown, fall back to current columns
    if expected_cols is None:
        expected_cols = list(X.columns)

    # Add any missing expected columns with zeros
    missing = [c for c in expected_cols if c not in X.columns]
    for c in missing:
        X[c] = 0

    # Drop unexpected columns
    X = X[[c for c in expected_cols]]

    return X


