"""Candidate-only calibration with date groups and a label-maturity purge.

This module never writes a serving pickle or changes the champion. Use a
frozen label definition, feature timestamps and a genuinely future test set
before considering a candidate for deployment.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def purged_date_folds(dates, label_end_dates, n_splits=4):
    dates = pd.to_datetime(dates).to_numpy()
    maturity = pd.to_datetime(label_end_dates).to_numpy()
    if len(dates) != len(maturity) or not len(dates):
        raise ValueError("Each input row needs its exact observed label maturity")
    if pd.isna(dates).any() or pd.isna(maturity).any():
        raise ValueError("Missing dates or label maturity cannot be calibrated")
    if np.any(maturity < dates):
        raise ValueError("Label maturity precedes prediction date")
    unique_dates = np.unique(dates)
    folds, audit = [], []
    for train_days, cal_days in TimeSeriesSplit(n_splits=n_splits).split(unique_dates):
        cal_start = unique_dates[cal_days[0]]
        train = np.flatnonzero(np.isin(dates, unique_dates[train_days]) & (maturity < cal_start))
        calibrate = np.flatnonzero(np.isin(dates, unique_dates[cal_days]))
        if not len(train) or not len(calibrate):
            raise ValueError("Insufficient matured history for a purged fold")
        if set(dates[train]) & set(dates[calibrate]):
            raise AssertionError("A date occurs in both fitting and calibration")
        folds.append((train, calibrate))
        audit.append({"fit_rows": len(train), "calibration_rows": len(calibrate),
                      "latest_fit_date": str(pd.Timestamp(dates[train].max()).date()),
                      "latest_fit_maturity": str(pd.Timestamp(maturity[train].max()).date()),
                      "calibration_start": str(pd.Timestamp(cal_start).date())})
    return folds, audit


def fit_linear_candidate(features, classes, dates, label_end_dates, n_splits=4, c=1.0):
    """Fit scaler and classifier inside each purged fold, never before it."""
    folds, audit = purged_date_folds(dates, label_end_dates, n_splits)
    y = np.asarray(classes)
    expected = set(y)
    for fit, cal in folds:
        if set(y[fit]) != expected or set(y[cal]) != expected:
            raise ValueError("A fold lacks a class; collect more dates rather than relax the purge")
    base = make_pipeline(StandardScaler(), LogisticRegression(
        C=c, class_weight="balanced", max_iter=1000, random_state=42))
    model = CalibratedClassifierCV(base, method="sigmoid", cv=folds, ensemble=True)
    model.fit(features, y)
    return model, audit
