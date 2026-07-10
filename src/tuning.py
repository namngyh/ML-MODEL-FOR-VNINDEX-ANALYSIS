"""Lightweight, leakage-aware hyperparameter search for VNIndex models."""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.metrics import classification_metrics, financial_metrics, regression_metrics
from src.models import fit_predict_hmm, fit_predict_supervised, model_specs


LIGHT_SEARCH_SPACE = {
    "SVC": [
        {"C": 1.0, "gamma": "scale"},
        {"C": 0.5, "gamma": "scale"},
        {"C": 3.0, "gamma": "scale"},
        {"C": 3.0, "gamma": 0.03},
    ],
    "SVR": [
        {"C": 5.0, "epsilon": 0.001, "gamma": "scale"},
        {"C": 1.0, "epsilon": 0.003, "gamma": "scale"},
        {"C": 10.0, "epsilon": 0.001, "gamma": "scale"},
        {"C": 5.0, "epsilon": 0.005, "gamma": 0.03},
    ],
    "Random Forest": [
        {"n_estimators": 350, "max_depth": 7, "min_samples_leaf": 20},
        {"n_estimators": 280, "max_depth": 5, "min_samples_leaf": 30, "max_features": "sqrt"},
        {"n_estimators": 350, "max_depth": 10, "min_samples_leaf": 12, "max_features": 0.7},
        {"n_estimators": 450, "max_depth": None, "min_samples_leaf": 25, "max_features": 0.5},
    ],
    "XGBoost": [
        {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.035, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 3.0},
        {"n_estimators": 320, "max_depth": 3, "learning_rate": 0.025, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 4.0},
        {"n_estimators": 220, "max_depth": 4, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 5.0},
    ],
    "LightGBM": [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.025, "num_leaves": 15, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
        {"n_estimators": 240, "max_depth": 3, "learning_rate": 0.04, "num_leaves": 7, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 3.0},
        {"n_estimators": 360, "max_depth": 5, "learning_rate": 0.02, "num_leaves": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 4.0},
        {"n_estimators": 260, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 24, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 5.0},
    ],
    "CatBoost": [
        {"iterations": 280, "depth": 4, "learning_rate": 0.035},
        {"iterations": 220, "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 4.0},
        {"iterations": 360, "depth": 4, "learning_rate": 0.025, "l2_leaf_reg": 5.0},
        {"iterations": 260, "depth": 5, "learning_rate": 0.035, "l2_leaf_reg": 6.0},
    ],
    "HMM Regime": [
        {"n_components": 4, "covariance_type": "diag", "n_iter": 400},
        {"n_components": 3, "covariance_type": "diag", "n_iter": 350},
        {"n_components": 5, "covariance_type": "diag", "n_iter": 450},
        {"n_components": 4, "covariance_type": "tied", "n_iter": 400},
    ],
}


def _bounded_component(value, transform="correlation"):
    if value is None or not np.isfinite(value):
        return 0.5
    if transform == "sharpe":
        return 0.5 + 0.5 * np.tanh(value / 2.0)
    return 0.5 + 0.5 * np.clip(value, -1.0, 1.0)


def _fold_score(cls, reg, fin):
    return (
        0.45 * cls["balanced_accuracy"]
        + 0.20 * cls["f1"]
        + 0.20 * _bounded_component(reg.get("spearman_ic"))
        + 0.15 * _bounded_component(fin.get("strategy_sharpe"), transform="sharpe")
    )


def _evaluate_candidate(name, params, frame, feature_cols, horizon, splits, random_state):
    target_return = f"future_return_{horizon}d"
    target_up = f"future_up_{horizon}d"
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        fold_train = frame.iloc[train_idx]
        fold_valid = frame.iloc[valid_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "HMM Regime":
                bundle = fit_predict_hmm(
                    fold_train,
                    fold_valid,
                    feature_cols,
                    horizon,
                    random_state=random_state,
                    hmm_params=params,
                )
            else:
                spec = model_specs(random_state=random_state, overrides={name: params})[name]
                bundle = fit_predict_supervised(
                    name,
                    spec,
                    fold_train[feature_cols],
                    fold_train[target_return],
                    fold_train[target_up].astype(int),
                    fold_valid[feature_cols],
                )

        cls = classification_metrics(
            fold_valid[target_up].astype(int), bundle.pred_direction, bundle.score_up
        )
        reg = regression_metrics(fold_valid[target_return], bundle.pred_return)
        strategy_return = bundle.pred_direction * fold_valid["daily_return_next"].fillna(0).to_numpy()
        fin = financial_metrics(
            fold_valid["date"],
            strategy_return,
            fold_valid["daily_return_next"].fillna(0),
            bundle.pred_direction,
        )
        fold_rows.append(
            {
                "fold": fold,
                "cv_score": _fold_score(cls, reg, fin),
                "balanced_accuracy": cls["balanced_accuracy"],
                "f1": cls["f1"],
                "spearman_ic": reg.get("spearman_ic", np.nan),
                "strategy_sharpe": fin.get("strategy_sharpe", np.nan),
            }
        )
    return pd.DataFrame(fold_rows)


def tune_horizon(frame, feature_cols, horizon, random_state=42, n_splits=3):
    """Return per-model overrides selected only from pre-test chronological data."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    splits = list(splitter.split(frame))
    trial_rows = []
    best_rows = []
    overrides = {}
    hmm_params = None

    for name, candidates in LIGHT_SEARCH_SPACE.items():
        model_trials = []
        for candidate_id, params in enumerate(candidates):
            started = time.perf_counter()
            error = ""
            try:
                folds = _evaluate_candidate(
                    name, params, frame, feature_cols, horizon, splits, random_state
                )
                aggregate = folds.mean(numeric_only=True).to_dict()
                score_std = folds["cv_score"].std(ddof=0)
            except Exception as exc:
                aggregate = {
                    "cv_score": -np.inf,
                    "balanced_accuracy": np.nan,
                    "f1": np.nan,
                    "spearman_ic": np.nan,
                    "strategy_sharpe": np.nan,
                }
                score_std = np.nan
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "horizon": horizon,
                "model": name,
                "candidate_id": candidate_id,
                "is_baseline_candidate": candidate_id == 0,
                "params_json": json.dumps(params, sort_keys=True),
                "cv_score": aggregate["cv_score"],
                "cv_score_std": score_std,
                "cv_balanced_accuracy": aggregate["balanced_accuracy"],
                "cv_f1": aggregate["f1"],
                "cv_spearman_ic": aggregate["spearman_ic"],
                "cv_strategy_sharpe": aggregate["strategy_sharpe"],
                "fit_seconds": time.perf_counter() - started,
                "error": error,
            }
            trial_rows.append(row)
            model_trials.append((row, params))

        best_trial, best_params = max(model_trials, key=lambda item: item[0]["cv_score"])
        best_trial["selected"] = True
        best_rows.append({**best_trial, "selected_params": best_trial["params_json"]})
        if name == "HMM Regime":
            hmm_params = best_params
        else:
            overrides[name] = best_params
        print(
            f"  {name}: candidate {best_trial['candidate_id']} "
            f"(CV score={best_trial['cv_score']:.4f}, {sum(row['fit_seconds'] for row, _ in model_trials):.1f}s)"
        )

    selected_keys = {(row["horizon"], row["model"], row["candidate_id"]) for row in best_rows}
    for row in trial_rows:
        row["selected"] = (row["horizon"], row["model"], row["candidate_id"]) in selected_keys

    return overrides, hmm_params, pd.DataFrame(trial_rows), pd.DataFrame(best_rows)
