import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from src.data import load_vnindex_csv
from src.features import chronological_split, make_features
from src.metrics import classification_metrics, financial_metrics, regression_metrics
from src.models import (
    feature_importance_rows,
    fit_predict_hmm,
    fit_predict_macd,
    fit_predict_supervised,
    model_specs,
)
from src.plots import (
    plot_baseline_vs_tuned,
    plot_cv_search_scores,
    plot_equity_curves,
    plot_feature_importance,
    plot_forecast_panel,
    plot_future_consensus,
    plot_future_model_heatmap,
    plot_future_price_targets,
    plot_future_return_forecast,
    plot_metric_heatmap,
    plot_price_macd,
    plot_tuning_delta_heatmaps,
    setup_plot_style,
)
from src.report import write_readme
from src.tuning import tune_horizon


def _split_summary(train, valid, test):
    rows = []
    for name, frame in [("train", train), ("valid", valid), ("test", test)]:
        rows.append(
            {
                "split": name,
                "rows": len(frame),
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _rank(metrics: pd.DataFrame, financial: pd.DataFrame):
    merged = metrics.merge(financial, on=["horizon", "model"], how="left")
    score_cols = ["balanced_accuracy", "f1", "spearman_ic", "r2", "strategy_sharpe"]
    ranked_parts = []
    for horizon, group in merged.groupby("horizon"):
        scored = group.copy()
        ic_score = 0.5 + 0.5 * scored["spearman_ic"].fillna(0).clip(-1, 1)
        sharpe_score = 0.5 + 0.5 * np.tanh(
            scored["strategy_sharpe"].fillna(0) / 2.0
        )
        scored["composite_score"] = (
            0.45 * scored["balanced_accuracy"]
            + 0.20 * scored["f1"]
            + 0.20 * ic_score
            + 0.15 * sharpe_score
        )
        ranks = []
        for col in score_cols:
            values = scored[col].replace([np.inf, -np.inf], np.nan)
            if values.notna().sum() <= 1:
                ranks.append(pd.Series(0.0, index=scored.index))
            else:
                ranks.append(values.rank(pct=True).fillna(0.0))
        scored["rank_score"] = pd.concat(ranks, axis=1).mean(axis=1)
        ranked_parts.append(scored.sort_values("rank_score", ascending=False))
    return pd.concat(ranked_parts, ignore_index=True)


def _make_prediction_rows(test, bundle, horizon):
    signal = pd.Series(bundle.pred_direction, index=test.index).astype(int)
    daily = test["daily_return_next"].fillna(0)
    strategy_return = signal * daily
    pred_return = bundle.pred_return
    if pred_return is None:
        pred_return = np.where(bundle.pred_direction == 1, test[f"future_return_{horizon}d"].mean(), 0.0)
    return pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "horizon": horizon,
            "model": bundle.model,
            "actual_return": test[f"future_return_{horizon}d"].to_numpy(),
            "actual_direction": test[f"future_up_{horizon}d"].astype(int).to_numpy(),
            "pred_return": pred_return,
            "pred_direction": bundle.pred_direction.astype(int),
            "score_up": bundle.score_up if bundle.score_up is not None else np.nan,
            "daily_return_next": daily.to_numpy(),
            "strategy_return": strategy_return.to_numpy(),
            "buy_hold_return": daily.to_numpy(),
        }
    )


def _build_future_forecasts(
    full_df,
    feature_cols,
    specs_by_horizon,
    hmm_params_by_horizon,
    ranking,
    horizons,
):
    latest = full_df.dropna(subset=feature_cols).iloc[[-1]].copy()
    latest_date = latest["date"].iloc[0]
    latest_close = latest["close"].iloc[0]
    rows = []
    regime_rows = []

    for horizon in horizons:
        specs = specs_by_horizon[horizon]
        target_return = f"future_return_{horizon}d"
        target_up = f"future_up_{horizon}d"
        train_df = full_df.dropna(subset=feature_cols + [target_return]).copy()
        x_train = train_df[feature_cols]
        y_train_ret = train_df[target_return]
        y_train_up = train_df[target_up].astype(int)
        x_future = latest[feature_cols]

        bundles = [fit_predict_macd(train_df, latest, horizon)]
        for name, spec in specs.items():
            bundles.append(fit_predict_supervised(name, spec, x_train, y_train_ret, y_train_up, x_future))
        hmm_bundle = fit_predict_hmm(
            train_df,
            latest,
            feature_cols,
            horizon,
            hmm_params=hmm_params_by_horizon[horizon],
        )
        bundles.append(hmm_bundle)

        for bundle in bundles:
            pred_return = float(np.asarray(bundle.pred_return)[0])
            pred_direction = int(np.asarray(bundle.pred_direction)[0])
            score_up = float(np.asarray(bundle.score_up)[0]) if bundle.score_up is not None else np.nan
            rank_match = ranking[(ranking["horizon"] == horizon) & (ranking["model"] == bundle.model)]
            rank_data = rank_match.iloc[0].to_dict() if len(rank_match) else {}
            current_regime = np.nan
            if bundle.model == "HMM Regime":
                current_regime = int(bundle.extra["predicted_states"][0])
                regime_rows.append(
                    {
                        "horizon": horizon,
                        "as_of_date": latest_date.date().isoformat(),
                        "current_regime": current_regime,
                        "regime_expected_return": pred_return,
                        "regime_state_mean_map": bundle.extra["state_mean_return"],
                    }
                )
            rows.append(
                {
                    "as_of_date": latest_date.date().isoformat(),
                    "latest_close": latest_close,
                    "horizon": horizon,
                    "target_date": (latest_date + pd.offsets.BDay(horizon)).date().isoformat(),
                    "model": bundle.model,
                    "pred_return": pred_return,
                    "pred_direction": pred_direction,
                    "direction_label": "Bullish" if pred_direction == 1 else "Bearish/Flat",
                    "score_up": score_up,
                    "predicted_close": latest_close * (1 + pred_return),
                    "rank_score": rank_data.get("rank_score", np.nan),
                    "test_balanced_accuracy": rank_data.get("balanced_accuracy", np.nan),
                    "test_f1": rank_data.get("f1", np.nan),
                    "test_spearman_ic": rank_data.get("spearman_ic", np.nan),
                    "test_strategy_sharpe": rank_data.get("strategy_sharpe", np.nan),
                    "current_regime": current_regime,
                }
            )

    future = pd.DataFrame(rows)
    consensus_rows = []
    for horizon, group in future.groupby("horizon"):
        weights = group["rank_score"].fillna(0).clip(lower=0)
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=group.index)
        weighted_return = np.average(group["pred_return"], weights=weights)
        bullish_share = group["pred_direction"].mean()
        median_return = group["pred_return"].median()
        if bullish_share >= 0.625 and median_return > 0:
            view = "Bullish"
            note = "Đa số mô hình ủng hộ xu hướng tăng."
        elif bullish_share <= 0.375 and median_return < 0:
            view = "Bearish"
            note = "Đa số mô hình nghiêng về rủi ro giảm."
        else:
            view = "Mixed/Neutral"
            note = "Tín hiệu phân hóa, nên ưu tiên quan sát xác nhận."
        consensus_rows.append(
            {
                "as_of_date": group["as_of_date"].iloc[0],
                "latest_close": group["latest_close"].iloc[0],
                "horizon": horizon,
                "target_date": group["target_date"].iloc[0],
                "models": len(group),
                "bullish_models": int(group["pred_direction"].sum()),
                "bearish_or_flat_models": int((1 - group["pred_direction"]).sum()),
                "bullish_share": bullish_share,
                "mean_pred_return": group["pred_return"].mean(),
                "median_pred_return": median_return,
                "weighted_pred_return": weighted_return,
                "median_predicted_close": group["predicted_close"].median(),
                "weighted_predicted_close": group["latest_close"].iloc[0] * (1 + weighted_return),
                "consensus_view": view,
                "interpretation": note,
            }
        )
    return future, pd.DataFrame(consensus_rows), pd.DataFrame(regime_rows)


def _compare_variants(baseline_ranking, tuned_ranking, best_parameters):
    metric_cols = [
        "rank_score",
        "composite_score",
        "balanced_accuracy",
        "f1",
        "spearman_ic",
        "r2",
        "strategy_sharpe",
        "strategy_total_return",
        "strategy_max_drawdown",
    ]
    baseline = baseline_ranking[["horizon", "model", *metric_cols]].rename(
        columns={col: f"baseline_{col}" for col in metric_cols}
    )
    tuned = tuned_ranking[["horizon", "model", *metric_cols]].rename(
        columns={col: f"tuned_{col}" for col in metric_cols}
    )
    comparison = baseline.merge(tuned, on=["horizon", "model"], how="outer")
    for col in metric_cols:
        comparison[f"delta_{col}"] = comparison[f"tuned_{col}"] - comparison[f"baseline_{col}"]
    comparison["improved_rank_score"] = comparison["delta_rank_score"] > 0
    comparison["improved_composite_score"] = comparison["delta_composite_score"] > 1e-12
    selected = best_parameters[
        [
            "horizon",
            "model",
            "candidate_id",
            "is_baseline_candidate",
            "selected_params",
            "cv_score",
            "cv_score_std",
            "fit_seconds",
        ]
    ]
    return comparison.merge(selected, on=["horizon", "model"], how="left")


def run_benchmark(data_path: Path, output_dir: Path, horizons=(5, 20, 60)):
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)

    raw = load_vnindex_csv(data_path)
    full_df, feature_cols = make_features(raw, horizons)
    historical_df = full_df.dropna(
        subset=feature_cols + [f"future_return_{h}d" for h in horizons] + ["daily_return_next"]
    ).copy()
    train, valid, test = chronological_split(historical_df)
    split_summary = _split_summary(train, valid, test)

    train_valid = pd.concat([train, valid], ignore_index=True)
    baseline_specs = model_specs()
    tuned_specs_by_horizon = {}
    hmm_params_by_horizon = {}
    prediction_frames = []
    baseline_prediction_frames = []
    metric_rows = []
    baseline_metric_rows = []
    financial_rows = []
    baseline_financial_rows = []
    importance_rows = []
    regime_rows = []
    tuning_trial_frames = []
    best_parameter_frames = []

    for horizon in horizons:
        print(f"Tuning horizon {horizon} phien...")
        overrides, hmm_params, trials, best_params = tune_horizon(
            train_valid,
            feature_cols,
            horizon,
        )
        tuned_specs = model_specs(overrides=overrides)
        tuned_specs_by_horizon[horizon] = tuned_specs
        hmm_params_by_horizon[horizon] = hmm_params
        tuning_trial_frames.append(trials)
        best_parameter_frames.append(best_params)

        target_return = f"future_return_{horizon}d"
        target_up = f"future_up_{horizon}d"
        x_train = train_valid[feature_cols]
        y_train_ret = train_valid[target_return]
        y_train_up = train_valid[target_up].astype(int)
        x_test = test[feature_cols]
        y_test_ret = test[target_return]
        y_test_up = test[target_up].astype(int)

        baseline_bundles = [fit_predict_macd(train_valid, test, horizon)]
        for name, spec in baseline_specs.items():
            baseline_bundles.append(
                fit_predict_supervised(name, spec, x_train, y_train_ret, y_train_up, x_test)
            )
        baseline_bundles.append(fit_predict_hmm(train_valid, test, feature_cols, horizon))

        tuned_bundles = [fit_predict_macd(train_valid, test, horizon)]
        for name, spec in tuned_specs.items():
            tuned_bundles.append(
                fit_predict_supervised(name, spec, x_train, y_train_ret, y_train_up, x_test)
            )
        hmm_bundle = fit_predict_hmm(
            train_valid,
            test,
            feature_cols,
            horizon,
            hmm_params=hmm_params,
        )
        tuned_bundles.append(hmm_bundle)
        for state, mean_return in hmm_bundle.extra["state_mean_return"].items():
            regime_rows.append({"horizon": horizon, "state": state, "mean_forward_return": mean_return})

        variants = [
            (
                baseline_bundles,
                baseline_prediction_frames,
                baseline_metric_rows,
                baseline_financial_rows,
                False,
            ),
            (tuned_bundles, prediction_frames, metric_rows, financial_rows, True),
        ]
        for bundles, pred_store, metric_store, financial_store, is_tuned in variants:
            for bundle in bundles:
                preds = _make_prediction_rows(test, bundle, horizon)
                pred_store.append(preds)
                cls = classification_metrics(y_test_up, bundle.pred_direction, bundle.score_up)
                reg = regression_metrics(y_test_ret, bundle.pred_return)
                metric_store.append({"horizon": horizon, "model": bundle.model, **cls, **reg})
                fin = financial_metrics(
                    preds["date"],
                    preds["strategy_return"],
                    preds["buy_hold_return"],
                    preds["pred_direction"],
                )
                financial_store.append({"horizon": horizon, "model": bundle.model, **fin})
                if is_tuned:
                    importance_rows.extend(feature_importance_rows(bundle, feature_cols, horizon))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    baseline_predictions = pd.concat(baseline_prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    baseline_metrics = pd.DataFrame(baseline_metric_rows)
    financial = pd.DataFrame(financial_rows)
    baseline_financial = pd.DataFrame(baseline_financial_rows)
    feature_importance = pd.DataFrame(importance_rows)
    regime_summary = pd.DataFrame(regime_rows)
    tuning_trials = pd.concat(tuning_trial_frames, ignore_index=True)
    best_parameters = pd.concat(best_parameter_frames, ignore_index=True)
    ranking = _rank(metrics, financial)
    baseline_ranking = _rank(baseline_metrics, baseline_financial)
    tuning_comparison = _compare_variants(baseline_ranking, ranking, best_parameters)
    future_forecasts, future_consensus, current_regime = _build_future_forecasts(
        full_df,
        feature_cols,
        tuned_specs_by_horizon,
        hmm_params_by_horizon,
        ranking,
        horizons,
    )

    raw.to_csv(output_dir / "clean_vnindex_data.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    baseline_predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    baseline_metrics.to_csv(output_dir / "baseline_metrics_by_horizon.csv", index=False)
    financial.to_csv(output_dir / "financial_metrics_by_horizon.csv", index=False)
    baseline_financial.to_csv(output_dir / "baseline_financial_metrics_by_horizon.csv", index=False)
    ranking.to_csv(output_dir / "model_ranking.csv", index=False)
    baseline_ranking.to_csv(output_dir / "baseline_model_ranking.csv", index=False)
    tuning_trials.to_csv(output_dir / "tuning_trials.csv", index=False)
    best_parameters.to_csv(output_dir / "best_hyperparameters.csv", index=False)
    tuning_comparison.to_csv(output_dir / "tuning_comparison.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    future_forecasts.to_csv(output_dir / "future_forecasts.csv", index=False)
    future_consensus.to_csv(output_dir / "future_consensus.csv", index=False)
    current_regime.to_csv(output_dir / "current_regime_forecast.csv", index=False)

    setup_plot_style()
    plot_price_macd(full_df.dropna(subset=feature_cols), figures_dir / "01_price_macd_rsi.png")
    plot_metric_heatmap(metrics, "balanced_accuracy", figures_dir / "02_balanced_accuracy_heatmap.png")
    plot_metric_heatmap(financial, "strategy_sharpe", figures_dir / "03_strategy_sharpe_heatmap.png")
    for horizon in horizons:
        plot_equity_curves(predictions, horizon, figures_dir / f"04_equity_curves_{horizon}d.png")
        plot_forecast_panel(predictions, horizon, figures_dir / f"05_forecast_panel_{horizon}d.png")
    plot_feature_importance(feature_importance, figures_dir / "06_feature_importance.png")
    plot_future_return_forecast(future_forecasts, figures_dir / "07_future_return_forecast.png")
    plot_future_price_targets(future_forecasts, future_consensus, figures_dir / "08_future_price_targets.png")
    plot_future_consensus(future_forecasts, future_consensus, figures_dir / "09_future_consensus_dashboard.png")
    plot_future_model_heatmap(future_forecasts, figures_dir / "10_future_model_heatmap.png")
    plot_tuning_delta_heatmaps(
        tuning_comparison, figures_dir / "11_tuning_delta_heatmaps.png"
    )
    plot_cv_search_scores(tuning_trials, figures_dir / "12_cv_search_scores.png")
    plot_baseline_vs_tuned(
        tuning_comparison, figures_dir / "13_baseline_vs_tuned_composite_score.png"
    )

    artifacts = {
        "price_macd": "outputs/figures/01_price_macd_rsi.png",
        "balanced_accuracy_heatmap": "outputs/figures/02_balanced_accuracy_heatmap.png",
        "sharpe_heatmap": "outputs/figures/03_strategy_sharpe_heatmap.png",
        "equity_5": "outputs/figures/04_equity_curves_5d.png",
        "equity_20": "outputs/figures/04_equity_curves_20d.png",
        "equity_60": "outputs/figures/04_equity_curves_60d.png",
        "forecast_5": "outputs/figures/05_forecast_panel_5d.png",
        "forecast_20": "outputs/figures/05_forecast_panel_20d.png",
        "forecast_60": "outputs/figures/05_forecast_panel_60d.png",
        "feature_importance": "outputs/figures/06_feature_importance.png",
        "future_return": "outputs/figures/07_future_return_forecast.png",
        "future_price_targets": "outputs/figures/08_future_price_targets.png",
        "future_consensus": "outputs/figures/09_future_consensus_dashboard.png",
        "future_heatmap": "outputs/figures/10_future_model_heatmap.png",
        "tuning_deltas": "outputs/figures/11_tuning_delta_heatmaps.png",
        "cv_search": "outputs/figures/12_cv_search_scores.png",
        "baseline_vs_tuned": "outputs/figures/13_baseline_vs_tuned_composite_score.png",
    }
    write_readme(
        Path("README.md"),
        {
            "rows": len(raw),
            "start": raw["date"].min().date().isoformat(),
            "end": raw["date"].max().date().isoformat(),
        },
        split_summary,
        metrics,
        financial,
        ranking,
        future_forecasts,
        future_consensus,
        baseline_ranking,
        tuning_trials,
        best_parameters,
        tuning_comparison,
        artifacts,
    )

    print("Done. Key artifacts:")
    print(f"- README.md")
    print(f"- {output_dir / 'model_ranking.csv'}")
    print(f"- {output_dir / 'metrics_by_horizon.csv'}")
    print(f"- {output_dir / 'financial_metrics_by_horizon.csv'}")
    print(f"- {output_dir / 'best_hyperparameters.csv'}")
    print(f"- {output_dir / 'tuning_comparison.csv'}")
    print(f"- {figures_dir}")
