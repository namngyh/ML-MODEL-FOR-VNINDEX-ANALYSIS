from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from catboost import CatBoostClassifier, CatBoostRegressor
from hmmlearn.hmm import GaussianHMM
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class PredictionBundle:
    model: str
    pred_direction: np.ndarray
    pred_return: np.ndarray | None
    score_up: np.ndarray | None
    extra: dict


def model_specs(random_state=42, overrides=None):
    overrides = overrides or {}

    def params(name, defaults):
        return {**defaults, **overrides.get(name, {})}

    return {
        "SVC": {
            "kind": "classifier",
            "estimator": make_pipeline(
                StandardScaler(),
                SVC(
                    **params(
                        "SVC",
                        {
                            "C": 1.0,
                            "gamma": "scale",
                            "class_weight": "balanced",
                            "probability": True,
                            "random_state": random_state,
                        },
                    )
                ),
            ),
        },
        "SVR": {
            "kind": "regressor",
            "estimator": make_pipeline(
                StandardScaler(),
                SVR(**params("SVR", {"C": 5.0, "epsilon": 0.001, "gamma": "scale"})),
            ),
        },
        "Random Forest": {
            "kind": "both",
            "classifier": RandomForestClassifier(
                **params(
                    "Random Forest",
                    {
                        "n_estimators": 350,
                        "max_depth": 7,
                        "min_samples_leaf": 20,
                        "class_weight": "balanced_subsample",
                        "random_state": random_state,
                        "n_jobs": -1,
                    },
                )
            ),
            "regressor": RandomForestRegressor(
                **{
                    key: value
                    for key, value in params(
                        "Random Forest",
                        {
                            "n_estimators": 350,
                            "max_depth": 7,
                            "min_samples_leaf": 20,
                            "random_state": random_state,
                            "n_jobs": -1,
                        },
                    ).items()
                    if key != "class_weight"
                }
            ),
        },
        "XGBoost": {
            "kind": "both",
            "classifier": XGBClassifier(
                **params(
                    "XGBoost",
                    {
                        "n_estimators": 250,
                        "max_depth": 3,
                        "learning_rate": 0.035,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                        "reg_lambda": 2.0,
                        "objective": "binary:logistic",
                        "eval_metric": "logloss",
                        "random_state": random_state,
                        "n_jobs": -1,
                    },
                )
            ),
            "regressor": XGBRegressor(
                **{
                    key: value
                    for key, value in params(
                        "XGBoost",
                        {
                            "n_estimators": 250,
                            "max_depth": 3,
                            "learning_rate": 0.035,
                            "subsample": 0.85,
                            "colsample_bytree": 0.85,
                            "reg_lambda": 2.0,
                            "objective": "reg:squarederror",
                            "random_state": random_state,
                            "n_jobs": -1,
                        },
                    ).items()
                    if key != "eval_metric"
                }
            ),
        },
        "LightGBM": {
            "kind": "both",
            "classifier": LGBMClassifier(
                **params(
                    "LightGBM",
                    {
                        "n_estimators": 300,
                        "max_depth": 4,
                        "learning_rate": 0.025,
                        "num_leaves": 15,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                        "reg_lambda": 2.0,
                        "class_weight": "balanced",
                        "random_state": random_state,
                        "n_jobs": -1,
                        "verbose": -1,
                    },
                )
            ),
            "regressor": LGBMRegressor(
                **{
                    key: value
                    for key, value in params(
                        "LightGBM",
                        {
                            "n_estimators": 300,
                            "max_depth": 4,
                            "learning_rate": 0.025,
                            "num_leaves": 15,
                            "subsample": 0.85,
                            "colsample_bytree": 0.85,
                            "reg_lambda": 2.0,
                            "random_state": random_state,
                            "n_jobs": -1,
                            "verbose": -1,
                        },
                    ).items()
                    if key != "class_weight"
                }
            ),
        },
        "CatBoost": {
            "kind": "both",
            "classifier": CatBoostClassifier(
                **params(
                    "CatBoost",
                    {
                        "iterations": 280,
                        "depth": 4,
                        "learning_rate": 0.035,
                        "loss_function": "Logloss",
                        "auto_class_weights": "Balanced",
                        "random_seed": random_state,
                        "verbose": False,
                        "allow_writing_files": False,
                    },
                )
            ),
            "regressor": CatBoostRegressor(
                **{
                    key: value
                    for key, value in params(
                        "CatBoost",
                        {
                            "iterations": 280,
                            "depth": 4,
                            "learning_rate": 0.035,
                            "loss_function": "RMSE",
                            "random_seed": random_state,
                            "verbose": False,
                            "allow_writing_files": False,
                        },
                    ).items()
                    if key != "auto_class_weights"
                }
            ),
        },
    }


def _probability(estimator, x):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(x)
        return 1 / (1 + np.exp(-score))
    return None


def fit_predict_supervised(name, spec, x_train, y_train_ret, y_train_up, x_test):
    kind = spec["kind"]
    pred_ret = None
    pred_dir = None
    score_up = None
    extra = {}

    if kind in {"classifier", "both"}:
        classifier = clone(spec.get("estimator") or spec["classifier"])
        classifier.fit(x_train, y_train_up)
        score_up = _probability(classifier, x_test)
        pred_dir = classifier.predict(x_test).astype(int)
        extra["classifier"] = classifier

    if kind in {"regressor", "both"}:
        regressor = clone(spec.get("estimator") or spec["regressor"])
        regressor.fit(x_train, y_train_ret)
        pred_ret = regressor.predict(x_test)
        extra["regressor"] = regressor
        if pred_dir is None:
            pred_dir = (pred_ret > 0).astype(int)
            score_up = 1 / (1 + np.exp(-pred_ret / (np.std(y_train_ret) + 1e-12)))

    if pred_ret is None and score_up is not None:
        scale = np.std(y_train_ret) if np.std(y_train_ret) > 0 else 1.0
        pred_ret = (score_up - 0.5) * 2 * scale

    return PredictionBundle(name, pred_dir, pred_ret, score_up, extra)


def fit_predict_macd(train_df, test_df, horizon):
    target = f"future_return_{horizon}d"
    mapping = train_df.groupby("macd_bullish")[target].mean().to_dict()
    fallback = train_df[target].mean()
    pred_dir = test_df["macd_bullish"].astype(int).to_numpy()
    pred_ret = np.array([mapping.get(int(signal), fallback) for signal in pred_dir])
    return PredictionBundle("MACD 12-26-9", pred_dir, pred_ret, pred_dir.astype(float), {"signal_return_map": mapping})


def fit_predict_hmm(train_df, test_df, feature_cols, horizon, random_state=42, hmm_params=None):
    hmm_features = [
        col
        for col in ["log_ret_1", "ret_5", "ret_20", "vol_20", "vol_60", "range_pct", "macd_hist", "rsi_14"]
        if col in feature_cols
    ]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[hmm_features])
    x_test = scaler.transform(test_df[hmm_features])
    params = {
        "n_components": 4,
        "covariance_type": "diag",
        "n_iter": 400,
        "random_state": random_state,
    }
    params.update(hmm_params or {})
    model = GaussianHMM(**params)
    model.fit(x_train)
    train_states = model.predict(x_train)
    test_states = model.predict(x_test)
    target = f"future_return_{horizon}d"
    state_mean = pd.DataFrame({"state": train_states, "target": train_df[target].to_numpy()}).groupby("state")[
        "target"
    ].mean()
    global_mean = train_df[target].mean()
    pred_ret = np.array([state_mean.get(state, global_mean) for state in test_states])
    pred_dir = (pred_ret > 0).astype(int)
    score_up = pd.Series(pred_ret).rank(pct=True).to_numpy()
    return PredictionBundle(
        "HMM Regime",
        pred_dir,
        pred_ret,
        score_up,
        {
            "model": model,
            "state_mean_return": state_mean.to_dict(),
            "features": hmm_features,
            "predicted_states": test_states,
        },
    )


def feature_importance_rows(bundle: PredictionBundle, feature_cols, horizon):
    estimator = bundle.extra.get("regressor") or bundle.extra.get("classifier")
    if estimator is None and "model" in bundle.extra:
        return []
    if hasattr(estimator, "named_steps"):
        estimator = list(estimator.named_steps.values())[-1]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return []
    order = np.argsort(values)[::-1][:20]
    return [
        {
            "horizon": horizon,
            "model": bundle.model,
            "feature": feature_cols[i],
            "importance": float(values[i]),
        }
        for i in order
    ]
