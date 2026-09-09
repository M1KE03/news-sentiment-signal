"""One function per numbered figure in section 8. No plotting code anywhere else.

Captions state the takeaway rather than describing the axes; each function
returns the Figure so a notebook can display it and `run_all.py` can save it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

STYLE = {
    "figure.figsize": (9, 5),
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
}

SCORER_LABEL = {"lm": "Loughran-McDonald", "vader": "VADER", "finbert": "FinBERT"}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig: plt.Figure, name: str, directory: Path = config.FIGURES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def figure_timestamp_audit(profiles: dict[str, pd.Series], verdicts: dict[str, str] | None = None) -> plt.Figure:
    """Stage 0: headlines by hour-of-day (ET), one panel per candidate dataset.

    The plan's first audit check, drawn. A date-only dump wearing a timestamp
    column shows as a single spike -- usually at midnight -- and is disqualified
    from D1 no matter how good the rest of it looks.

    `profiles` maps candidate name -> `audit.hour_histogram(df)`.
    """
    use_style()
    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    close_hour = config.MARKET_CLOSE_ET.hour
    for ax, (name, hist) in zip(axes, profiles.items()):
        share = hist / max(hist.sum(), 1)
        ax.bar(hist.index, share, width=0.85, color="#1f4e79")
        ax.axvline(9.5, color="#2e7d32", lw=1, ls="--")
        ax.axvline(close_hour, color="#c0392b", lw=1, ls="--")
        ax.annotate("open", xy=(9.5, ax.get_ylim()[1]), fontsize=7, color="#2e7d32",
                    ha="right", va="top", rotation=90)
        ax.annotate("close", xy=(close_hour, ax.get_ylim()[1]), fontsize=7, color="#c0392b",
                    ha="right", va="top", rotation=90)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("hour of day (ET)")
        title = name if verdicts is None else f"{name}\n{verdicts.get(name, '')[:60]}"
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("share of headlines")
    # P24: names what the panel plots, not what it is expected to show.
    fig.suptitle("Distribution of publication times within the day, by source", y=1.04)
    fig.tight_layout()
    return fig


def figure1_confusion(results: dict[str, dict]) -> plt.Figure:
    """Figure 1: confusion matrices, three scorers side by side."""
    use_style()
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results.items()):
        cm = np.asarray(r["confusion"], dtype=float)
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(r["labels"])), r["labels"], rotation=45, ha="right")
        ax.set_yticks(range(len(r["labels"])), r["labels"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9,
                )
        ax.set_title(f"{SCORER_LABEL.get(name, name)}\nmacro-F1 {r['macro_f1']:.3f}")
        ax.set_xlabel("predicted")
        ax.grid(False)
    axes[0].set_ylabel("true")
    # P24: the error structure is the result, so it cannot be in the title.
    fig.suptitle("Confusion matrices by scorer, on the evaluation set", y=1.02)
    fig.tight_layout()
    return fig


def figure2_lag_family(
    families: dict[str, pd.DataFrame],
    contemporaneous: dict[str, dict] | None = None,
    placebo: dict[str, float] | None = None,
) -> plt.Figure:
    """Figure 2 (headline): coefficient +/- NW CI by horizon, one panel per scorer.

    `families` maps scorer -> the `lag_family` table. `contemporaneous` optionally
    supplies {'coef','nw_se'} per scorer to draw the t=0 point. `placebo` maps
    scorer -> permutation p-value, annotated on h=1.
    """
    use_style()
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, fam) in zip(axes, families.items()):
        h = list(fam["horizon"])
        b = list(fam["coef"])
        se = list(fam["nw_se"])
        if contemporaneous and name in contemporaneous:
            h = [0] + h
            b = [contemporaneous[name]["coef"]] + b
            se = [contemporaneous[name]["nw_se"]] + se
        b, se = np.asarray(b), np.asarray(se)
        ax.errorbar(h, b, yerr=1.96 * se, fmt="o", capsize=4, lw=1.4, color="#1f4e79")
        ax.axhline(0, color="black", lw=1)
        ax.set_title(SCORER_LABEL.get(name, name))
        ax.set_xlabel("horizon (trading days)")
        ax.set_xticks(h)
        if placebo and name in placebo:
            ax.annotate(
                f"placebo p = {placebo[name]:.3f}",
                xy=(1, b[h.index(1)]),
                xytext=(6, 14),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set_ylabel(r"$\beta$ on $S_t$  (log return per unit sentiment)")
    # P24: "next-day nothing" asserted the primary result, and "same-day
    # association" asserts one that RQ2 suppression means will not be estimated
    # at all on this corpus.
    fig.suptitle(
        "Coefficients on $S_t$ by horizon, with 95% pointwise Newey-West intervals",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def figure3_dispersion_volume(panel: pd.DataFrame, scorer: str = "finbert") -> plt.Figure:
    """Figure 3: dispersion d_t against next-day detrended volume, with a fitted line."""
    use_style()
    df = panel[[f"d_{scorer}", "log_turnover_detrended_lead1"]].dropna()
    x = df[f"d_{scorer}"].to_numpy()
    y = df["log_turnover_detrended_lead1"].to_numpy()
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=8, alpha=0.35, color="#1f4e79", edgecolors="none")
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#c0392b", lw=2,
                label=f"slope = {slope:.3f}")
        ax.legend(frameon=False)
    ax.set_xlabel(f"within-day sentiment dispersion $d_t$ ({SCORER_LABEL.get(scorer, scorer)})")
    ax.set_ylabel("next-day detrended log turnover")
    # P24: this asserted RQ4's answer. It states the axes instead.
    ax.set_title("Next-session detrended volume against within-day tone dispersion")
    fig.tight_layout()
    return fig


def figure4_context(panel: pd.DataFrame, episodes: dict[str, str] | None = None,
                    scorer: str = "finbert", smooth: int = 21) -> plt.Figure:
    """Figure 4: S_t and SPY over the sample, episodes marked. Context, not evidence."""
    use_style()
    df = panel.dropna(subset=[f"s_{scorer}"]).copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["date"], df[f"s_{scorer}"].rolling(smooth, min_periods=1).mean(),
            color="#1f4e79", lw=1.4, label=f"$S_t$ ({smooth}-day mean)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"{SCORER_LABEL.get(scorer, scorer)} daily sentiment")
    ax.set_xlabel("")

    ax2 = ax.twinx()
    ax2.plot(df["date"], df["close_adj"] if "close_adj" in df else np.nan,
             color="#999999", lw=1.1, label="SPY")
    ax2.set_ylabel("SPY (adjusted close)")
    ax2.grid(False)

    for label, when in (episodes or {}).items():
        ts = pd.Timestamp(when)
        ax.axvline(ts, color="#c0392b", lw=0.9, ls="--", alpha=0.8)
        ax.annotate(label, xy=(ts, ax.get_ylim()[1]), rotation=90, fontsize=7,
                    va="top", ha="right", color="#c0392b")

    ax.set_title("Aggregate headline tone and the index level")
    fig.tight_layout()
    return fig


def figure_acf(panel: pd.DataFrame, scorer: str = "finbert", lags: int = 40) -> plt.Figure:
    """EDA support: the ACF of S_t, which is the written justification for D10 and D12."""
    use_style()
    from statsmodels.graphics.tsaplots import plot_acf

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(panel[f"s_{scorer}"].dropna(), lags=lags, ax=ax, zero=False)
    ax.set_title(f"Autocorrelation of $S_t$ ({SCORER_LABEL.get(scorer, scorer)})")
    ax.set_xlabel("lag (trading days)")
    fig.tight_layout()
    return fig


def figure_coverage(panel: pd.DataFrame) -> plt.Figure:
    """EDA support: n_t over time. Coverage drift is a named limitation, so plot it."""
    use_style()
    fig, ax = plt.subplots()
    ax.plot(panel["date"], panel["n_headlines"], lw=0.6, color="#1f4e79", alpha=0.6)
    ax.plot(panel["date"], panel["n_headlines"].rolling(63, min_periods=1).mean(),
            lw=2, color="#c0392b", label="63-day mean")
    ax.set_ylabel("headlines per trading day, $n_t$")
    ax.legend(frameon=False)
    # P24: whether coverage is non-stationary enough to matter is a finding.
    ax.set_title("Headlines per session over the sample window")
    fig.tight_layout()
    return fig
