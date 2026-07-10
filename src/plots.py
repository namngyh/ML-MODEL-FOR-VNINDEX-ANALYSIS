from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.size": 9,
        }
    )


def plot_price_macd(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    axes[0].plot(df["date"], df["close"], color="#1f4e79", linewidth=1.2, label="VNIndex Close")
    bullish = df["macd_bullish"] == 1
    axes[0].fill_between(df["date"], df["close"].min(), df["close"].max(), where=bullish, color="#d9ead3", alpha=0.35)
    axes[0].set_title("VNIndex va vung MACD bullish")
    axes[0].legend(loc="upper left")
    axes[1].plot(df["date"], df["macd"], label="MACD", color="#0b5394")
    axes[1].plot(df["date"], df["macd_signal"], label="Signal", color="#cc0000")
    axes[1].bar(df["date"], df["macd_hist"], color=np.where(df["macd_hist"] >= 0, "#6aa84f", "#e06666"), alpha=0.7)
    axes[1].legend(loc="upper left")
    axes[2].plot(df["date"], df["rsi_14"], color="#674ea7", label="RSI 14")
    axes[2].axhline(70, color="#cc0000", linestyle="--", linewidth=0.8)
    axes[2].axhline(30, color="#38761d", linestyle="--", linewidth=0.8)
    axes[2].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_heatmap(metrics: pd.DataFrame, metric: str, output_path: Path):
    pivot = metrics.pivot_table(index="model", columns="horizon", values=metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=pivot.stack().median(), ax=ax)
    ax.set_title(f"So sanh {metric} theo horizon")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_equity_curves(predictions: pd.DataFrame, horizon: int, output_path: Path):
    subset = predictions[predictions["horizon"] == horizon].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    for model, group in subset.groupby("model"):
        daily = group.sort_values("date")["strategy_return"].fillna(0)
        equity = (1 + daily).cumprod()
        ax.plot(group.sort_values("date")["date"], equity, label=model, linewidth=1.1)
    bh = subset[subset["model"] == subset["model"].iloc[0]].sort_values("date")
    ax.plot(bh["date"], (1 + bh["buy_hold_return"].fillna(0)).cumprod(), label="Buy & Hold", color="black", linewidth=1.6)
    ax.set_title(f"Equity curve tren tap test - horizon {horizon} phien")
    ax.set_ylabel("Growth of 1")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_forecast_panel(predictions: pd.DataFrame, horizon: int, output_path: Path):
    subset = predictions[predictions["horizon"] == horizon].copy()
    models = list(subset["model"].drop_duplicates())
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, nrows * 3)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, model in zip(axes, models):
        group = subset[subset["model"] == model].sort_values("date")
        ax.plot(group["date"], group["actual_return"], color="#444444", linewidth=0.9, label="Actual future return")
        ax.plot(group["date"], group["pred_return"], color="#0b5394", linewidth=0.9, label="Predicted return")
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.fill_between(
            group["date"],
            group["actual_return"].min(),
            group["actual_return"].max(),
            where=group["pred_direction"].astype(bool),
            color="#d9ead3",
            alpha=0.35,
        )
        ax.set_title(model)
    for ax in axes[len(models) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"Du bao return va tin hieu long/flat - horizon {horizon} phien", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path):
    if feature_importance.empty:
        return
    top = (
        feature_importance.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(18)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, y="feature", x="importance", color="#3c78d8", ax=ax)
    ax.set_title("Top feature importance trung binh qua cac mo hinh cay/boosting")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_future_return_forecast(future: pd.DataFrame, output_path: Path):
    data = future.copy()
    data["pred_return_pct"] = data["pred_return"] * 100
    data["direction"] = np.where(data["pred_direction"] == 1, "Bullish", "Bearish/Flat")
    g = sns.catplot(
        data=data,
        x="model",
        y="pred_return_pct",
        hue="direction",
        col="horizon",
        kind="bar",
        height=4,
        aspect=1.25,
        palette={"Bullish": "#38761d", "Bearish/Flat": "#cc0000"},
        sharey=False,
    )
    g.set_axis_labels("", "Predicted forward return (%)")
    g.set_titles("{col_name} phien")
    for ax in g.axes.flat:
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.tick_params(axis="x", rotation=35)
    g.fig.suptitle("Du bao return tu phien moi nhat theo tung mo hinh", y=1.05)
    g.fig.tight_layout()
    g.fig.savefig(output_path)
    plt.close(g.fig)


def plot_future_price_targets(future: pd.DataFrame, consensus: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    latest_close = future["latest_close"].iloc[0]
    ax.axhline(latest_close, color="#222222", linestyle="--", linewidth=1.1, label=f"Latest close {latest_close:,.2f}")
    for horizon, group in future.groupby("horizon"):
        group = group.sort_values("predicted_close")
        target_date = pd.to_datetime(group["target_date"].iloc[0])
        x_offsets = np.linspace(-0.18, 0.18, len(group))
        x_values = [target_date + pd.Timedelta(days=float(offset)) for offset in x_offsets]
        colors = np.where(group["pred_direction"] == 1, "#38761d", "#cc0000")
        ax.scatter(x_values, group["predicted_close"], s=55, c=colors, alpha=0.85, label=f"{horizon} phien models")
        median_target = consensus.loc[consensus["horizon"] == horizon, "median_predicted_close"].iloc[0]
        ax.scatter(target_date, median_target, s=180, marker="D", color="#0b5394", edgecolor="white", linewidth=0.8)
        ax.text(target_date, median_target, f"  median {horizon}d: {median_target:,.0f}", va="center", fontsize=9)
    ax.set_title("VNIndex projected close theo model va horizon")
    ax.set_ylabel("Projected close")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_future_consensus(future: pd.DataFrame, consensus: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=consensus, x="horizon", y="bullish_share", color="#6aa84f", ax=axes[0])
    axes[0].axhline(0.5, color="#666666", linestyle="--", linewidth=0.9)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Ty le mo hinh bullish")
    axes[0].set_ylabel("Bullish share")
    axes[0].set_xlabel("Horizon")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.0f%%", labels=[f"{v * 100:.0f}%" for v in consensus["bullish_share"]])

    plot_data = consensus.copy()
    plot_data["median_pred_return_pct"] = plot_data["median_pred_return"] * 100
    colors = np.where(plot_data["median_pred_return"] >= 0, "#38761d", "#cc0000")
    axes[1].bar(plot_data["horizon"].astype(str), plot_data["median_pred_return_pct"], color=colors)
    axes[1].axhline(0, color="#666666", linewidth=0.9)
    axes[1].set_title("Median predicted return")
    axes[1].set_ylabel("%")
    axes[1].set_xlabel("Horizon")
    fig.suptitle("Dashboard dong thuan du bao tu cac mo hinh")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_future_model_heatmap(future: pd.DataFrame, output_path: Path):
    pivot = future.pivot_table(index="model", columns="horizon", values="pred_return")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot * 100, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Heatmap predicted forward return (%)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_tuning_delta_heatmaps(comparison: pd.DataFrame, output_path: Path):
    metrics = [
        ("delta_balanced_accuracy", "Delta Balanced Accuracy"),
        ("delta_spearman_ic", "Delta Spearman IC"),
        ("delta_strategy_sharpe", "Delta Strategy Sharpe"),
    ]
    data = comparison[comparison["model"] != "MACD 12-26-9"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, (column, title) in zip(axes, metrics):
        pivot = data.pivot_table(index="model", columns="horizon", values=column)
        limit = np.nanmax(np.abs(pivot.to_numpy()))
        limit = max(limit, 0.01)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            vmin=-limit,
            vmax=limit,
            cbar=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("")
    fig.suptitle("Muc thay doi tren tap test: Tuned - Baseline")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cv_search_scores(trials: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, horizon in zip(axes, sorted(trials["horizon"].unique())):
        subset = trials[trials["horizon"] == horizon].copy()
        sns.stripplot(
            data=subset,
            x="model",
            y="cv_score",
            hue="selected",
            palette={False: "#9e9e9e", True: "#0b5394"},
            size=6,
            jitter=0.15,
            ax=ax,
        )
        ax.set_title(f"{horizon} phien")
        ax.tick_params(axis="x", rotation=35)
        ax.set_xlabel("")
        ax.set_ylabel("CV composite score" if ax is axes[0] else "")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.suptitle("Diem CV cua 4 cau hinh ung vien; mau xanh la tham so duoc chon")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_baseline_vs_tuned(comparison: pd.DataFrame, output_path: Path):
    data = comparison[comparison["model"] != "MACD 12-26-9"].copy()
    long = data.melt(
        id_vars=["horizon", "model"],
        value_vars=["baseline_composite_score", "tuned_composite_score"],
        var_name="variant",
        value_name="composite_score",
    )
    long["variant"] = long["variant"].map(
        {"baseline_composite_score": "Baseline", "tuned_composite_score": "Tuned"}
    )
    g = sns.catplot(
        data=long,
        x="model",
        y="composite_score",
        hue="variant",
        col="horizon",
        kind="bar",
        height=4.2,
        aspect=1.25,
        palette={"Baseline": "#9e9e9e", "Tuned": "#0b5394"},
        sharey=True,
    )
    g.set_axis_labels("", "Out-of-sample composite score")
    g.set_titles("{col_name} phien")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
    g.fig.suptitle("So sanh composite score tren tap test: Baseline va Tuned", y=1.05)
    g.fig.tight_layout()
    g.fig.savefig(output_path)
    plt.close(g.fig)
