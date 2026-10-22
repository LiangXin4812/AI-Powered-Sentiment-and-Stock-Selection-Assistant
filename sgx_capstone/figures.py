"""Figures for the final report, generated from the retained research tables.

    python -m sgx_capstone figures            -> outputs/figures/*.png + sources.json

Every chart reads only files under outputs/research/ (or the evidence folder)
and prints its source file in the bottom-left corner. sources.json records,
for each figure, the input files, their SHA-256 and the numbers quoted in the
report caption, so a caption can be checked against the chart that backs it.

Colours follow one fixed order (blue, orange, aqua) checked for colour-vision
deficiency; red is used only for "negative / fell", grey only for baselines.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch    # noqa: E402

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GREY, LIGHT, XL = "#898781", "#c3c2b7", "#e1e0d9"
INK, INK2 = "#0b0b0b", "#52514e"
BLUE_T, ORANGE_T, AQUA_T, NAVY_T = "#dbe8f8", "#fbe3d8", "#d6f1e6", "#1a2e55"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.8, "savefig.dpi": 300})


class Builder:
    def __init__(self, root: Path, out: Path | None = None):
        self.root = Path(root)
        self.res = self.root / "outputs" / "research"
        self.out = Path(out or self.root / "outputs" / "figures")
        self.out.mkdir(parents=True, exist_ok=True)
        self.sources = {}

    def read(self, name):
        return pd.read_csv(self.res / name)

    def _sha(self, rel):
        return hashlib.sha256((self.root / rel).read_bytes()).hexdigest()[:16]

    def save(self, fig, name, inputs, facts):
        label = ", ".join(Path(i).name for i in inputs)
        # Place the source line just below everything already drawn, so it
        # never collides with tick labels or a legend under the axes.
        renderer = fig.canvas.get_renderer()
        bb = fig.get_tightbbox(renderer)
        fig.text(bb.x0 / fig.get_figwidth(), (bb.y0 - 0.06) / fig.get_figheight(),
                 f"Source: {label}  ·  sgx_capstone/figures.py",
                 fontsize=6.3, color=GREY, ha="left", va="top")
        path = self.out / name
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.sources[name] = {"inputs": {i: self._sha(i) for i in inputs},
                              "facts": facts}
        return path


def _style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=XL, lw=0.7)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════════
# F1  architecture
# ═══════════════════════════════════════════════════════════════════════════

def _box(ax, x, y, w, h, title, sub=None, fill="white", edge=INK2, fs=7.8, sfs=6.4,
         tcolor=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=fill, ec=edge, lw=0.9))
    if sub:
        ax.text(x + w / 2, y + h - 0.02, title, ha="center", va="top", fontsize=fs,
                weight="bold", color=tcolor)
        ax.text(x + w / 2, y + (h - 0.03) / 2, sub, ha="center", va="center",
                fontsize=sfs, color=INK2 if tcolor == INK else tcolor, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                weight="bold", color=tcolor)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color=INK2, shrinkA=0, shrinkB=0))


def fig_architecture(b: Builder):
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.075, 0.99)
    ax.axis("off")
    W4, G = 0.235, 0.02
    xs = [0.0, W4 + G, 2 * (W4 + G), 3 * (W4 + G)]
    srcs = [("News feeds", "Google News, Business\nTimes, Straits Times"),
            ("Yahoo Finance", "daily prices for\n15 SGX tickers"),
            ("Macro sources", "SORA, CPI, VIX,\nUS yield spread"),
            ("Options proxy", "EWS (MSCI Singapore\nETF) option chain")]
    for x, (t, s) in zip(xs, srcs):
        _box(ax, x, 0.895, W4, 0.085, t, s, fill=XL, fs=7.2, sfs=6.0)
    tasks = [("T1 News, sentiment", "FinBERT, LexRank, event\ntags, political bucket"),
             ("T2 Technical", "RSI, MACD, Bollinger,\nVWAP, weekly trend"),
             ("T3 Macro regime", "6-rule priority cascade\nsector-impact narrative"),
             ("T4 Options flow", "put/call volume and\nopen interest")]
    for x, (t, s) in zip(xs, tasks):
        _box(ax, x, 0.765, W4, 0.09, t, s, fill=BLUE_T, edge=BLUE, fs=7.3, sfs=6.0)
        _arrow(ax, x + W4 / 2, 0.895, x + W4 / 2, 0.857)
    _box(ax, 0.0, 0.64, 1.0, 0.09, "PostgreSQL store (one row per ticker and date)",
         "news_articles, technical_signals, macro_indicators, market_options,\n"
         "llm_analyses, task6_predictions, task6_champion_log, final_reports",
         fill=XL, fs=7.7, sfs=6.3)
    for x in xs:
        _arrow(ax, x + W4 / 2, 0.765, x + W4 / 2, 0.732)
    HW = 0.49
    _box(ax, 0.0, 0.48, HW, 0.13, "T5 Evidence layer (what the day said)",
         "rule layer: ridge magnitudes, domain signs\nLogReg + RF second opinion with SHAP\n"
         "contradiction check, warning level\nLLM writer with six output guardrails",
         fill=ORANGE_T, edge=ORANGE, fs=7.7, sfs=6.3)
    _box(ax, 1 - HW, 0.48, HW, 0.13, "T6 Forecast layer (next 3 sessions)",
         "LogReg, RandomForest, XGBoost, 21 features\nout-of-fold sigmoid calibration\n"
         "champion serves, challengers logged\npreflight rejects a leaky artefact",
         fill=AQUA_T, edge=AQUA, fs=7.7, sfs=6.3)
    _arrow(ax, HW / 2, 0.64, HW / 2, 0.617)
    _arrow(ax, 1 - HW / 2, 0.64, 1 - HW / 2, 0.617)
    _box(ax, 0.0, 0.3, 1.0, 0.14, "T7 Report layer",
         "input builder with look-ahead guard; 7-year price context since 2019\n"
         "11 patterns (evidence vs price vs forecast) > attention level > withholding below the prior-data rule\n"
         "outlook: typical range and quieter-than-most ranking with its R14 record, explained by group Shapley\n"
         "LLM summarises the news (T5); report text checked R1-R7; deterministic fallback",
         fill=NAVY_T, edge=NAVY_T, fs=7.7, sfs=6.3, tcolor="white")
    _arrow(ax, HW / 2, 0.48, HW / 2, 0.442)
    _arrow(ax, 1 - HW / 2, 0.48, 1 - HW / 2, 0.442)
    _box(ax, 0.0, 0.15, HW, 0.105, "Daily Signal Report",
         "board of 15 stocks, stock cards,\nnext three sessions with record,\nreviewer appendix",
         fill=BLUE_T, edge=BLUE, fs=7.7, sfs=6.3)
    _box(ax, 1 - HW, 0.15, HW, 0.105, "Research Report",
         "reliability against baselines, calibration,\nresearch programme R1-R14,\nincident log, guideline map",
         fill=XL, fs=7.7, sfs=6.3)
    _arrow(ax, HW / 2, 0.3, HW / 2, 0.257)
    _arrow(ax, 1 - HW / 2, 0.3, 1 - HW / 2, 0.257)
    ax.text(0.5, 0.105, "config.yaml, settings and prompts are hashed into a SHA-256 manifest; "
            "the serving path is a Docker image\nwith pinned artefacts, a doctor command, "
            "offline tests and an offline sample report", ha="center", va="center",
            fontsize=6.5, color=INK2, style="italic")
    return b.save(fig, "f1_architecture.png", ["config.yaml"], {})


# ═══════════════════════════════════════════════════════════════════════════
# F2-F3  evidence layer
# ═══════════════════════════════════════════════════════════════════════════

def fig_net_score(b: Builder):
    r8 = b.read("R8_rl_v8_readings.csv")
    g = r8.groupby("date")
    dd = pd.DataFrame({"lo": g.net_rl_v7.min(), "hi": g.net_rl_v7.max(),
                       "med": g.net_rl_v7.median(), "med8": g.net_rl_v8.median(),
                       "lo8": g.net_rl_v8.min()}).reset_index()
    x = np.arange(len(dd))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.fill_between(x, dd.lo, dd.hi, color=BLUE_T, lw=0, label="Range across 15 stocks (rl_v7)")
    ax.plot(x, dd.med, color=BLUE, lw=1.8, marker="o", ms=3.2, label="Median stock, rl_v7")
    ax.plot(x, dd.lo, color=RED, lw=1.1, ls="--", marker="v", ms=3.2,
            label="Most negative stock, rl_v7")
    ax.plot(x, dd.med8, color=ORANGE, lw=1.8, label="Median stock, rl_v8 replay")
    for yv in (0.35, -0.35):
        ax.axhline(yv, color=INK2, lw=0.8, ls=(0, (6, 3)))
    ax.text(len(dd) - 0.6, 0.37, "positive line +0.35", ha="right", va="bottom", fontsize=7.4)
    ax.text(len(dd) - 0.6, -0.33, "negative line −0.35", ha="right", va="bottom", fontsize=7.4)
    ax.axhline(0, color=LIGHT, lw=0.7)
    ax.set_xticks(x, [d[5:] for d in dd.date], rotation=60, fontsize=7)
    ax.set_ylabel("Net rule score")
    ax.set_ylim(-0.45, 1.0)
    _style(ax)
    ax.legend(fontsize=7.2, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    facts = {"min_rl_v7": float(dd.lo.min()), "days_min_above_zero": int((dd.lo > 0).sum()),
             "days": int(len(dd))}
    return b.save(fig, "f2_net_score.png", ["outputs/research/R8_rl_v8_readings.csv"], facts)


def fig_rule_tilt(b: Builder):
    r7 = b.read("R7_t5_rule_audit.csv")
    r7["fam"] = r7.factor.where(~r7.factor.str.startswith("Bullish corporate events"),
                                "Bullish corporate events (all counts)")
    r7["fam"] = r7.fam.where(~r7.fam.str.startswith("Bearish corporate events"),
                             "Bearish corporate events")
    agg = r7.groupby("fam").agg(t=("tilt_contrib", "sum"), f=("fired%", "sum")).sort_values("t")
    agg = agg[agg.t.abs() >= 0.006]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cols = [ORANGE if "corporate events" in i and v > 0 else BLUE if v > 0 else RED
            for i, v in zip(agg.index, agg.t)]
    bars = ax.barh(agg.index, agg.t, color=cols, height=0.62)
    for bb, v, f in zip(bars, agg.t, agg.f):
        lab = f"{v:+.3f}  (fires {min(f, 1) * 100:.1f}%)" if f >= 0.99 else f"{v:+.3f}  (fires {f:.0%})"
        ax.text(v + (0.004 if v > 0 else -0.004), bb.get_y() + bb.get_height() / 2, lab,
                va="center", ha="left" if v > 0 else "right", fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Contribution to the mean net score (+0.381 overall); orange: the event rule")
    ax.set_xlim(-0.43, 0.38)
    ax.tick_params(axis="y", labelsize=7.4)
    _style(ax, "x")
    return b.save(fig, "f3_rule_tilt.png", ["outputs/research/R7_t5_rule_audit.csv"],
                  {"total_tilt": float(r7.tilt_contrib.sum())})


# ═══════════════════════════════════════════════════════════════════════════
# F4-F6  forecast layer
# ═══════════════════════════════════════════════════════════════════════════

def fig_calibration(b: Builder):
    r2 = b.read("R2_horizon_calibration.csv")
    g3 = r2[r2.h == 3]
    models = ["LogReg", "RandomF", "XGBoost"]
    lk = g3[g3.calib == "leaky"].set_index("model").loc[models]
    of = g3[g3.calib == "oof"].set_index("model").loc[models]
    neu = float(g3.neutral_acc.iloc[0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    xx, w = np.arange(3), 0.36
    for ax_, col, ttl, ylab in ((a1, "share_ge90", "Forecasts stated at ≥ 0.9", "share of forecasts"),
                                (a2, "acc", "Directional accuracy", "accuracy")):
        b1 = ax_.bar(xx - w / 2 - 0.01, lk[col], w, color=ORANGE, label="Original (prefit) calibration")
        b2 = ax_.bar(xx + w / 2 + 0.01, of[col], w, color=BLUE, label="Out-of-fold calibration")
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax_.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.0%}", ha="center", fontsize=7)
        ax_.set_xticks(xx, models)
        ax_.set_title(ttl, loc="left", fontsize=9, weight="bold")
        ax_.set_ylabel(ylab)
        _style(ax_)
    a1.set_ylim(0, 0.75)
    a2.set_ylim(0, 0.70)
    a2.axhline(neu, color=GREY, ls="--", lw=1.0)
    a2.text(-0.45, 0.655, f"dashed: always 'neutral' ({neu:.1%})", ha="left", fontsize=7, color=INK2)
    a1.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    facts = {"share_ge90_leaky": lk.share_ge90.round(3).to_dict(),
             "acc_oof": of.acc.round(3).to_dict(), "always_neutral": neu,
             "brier_leaky": lk.brier.round(3).to_dict(), "brier_oof": of.brier.round(3).to_dict()}
    return b.save(fig, "f4_calibration.png", ["outputs/research/R2_horizon_calibration.csv"], facts)


def fig_weekly(b: Builder):
    """Left: the live champion record (what readers saw). Right: the same
    weekly view for the honestly calibrated walk-forward model."""
    live = b.read("R10_suppression_replay.csv")
    live["week"] = pd.to_datetime(live.date).dt.strftime("%G-W%V")
    lw = live.groupby("week")[["pred", "actual"]].apply(lambda t: pd.Series({
        "acc": (t.pred == t.actual).mean(), "bear": (t.actual == "bearish").mean(),
        "neu": (t.actual == "neutral").mean(), "pbull": (t.pred == "bullish").mean(),
        "n": len(t)})).reset_index()
    wf = b.read("walk_forward_predictions.csv")
    s_ = wf[(wf.h == 3) & (wf.calib == "oof") & (wf.model == "RandomF")]
    wk = s_.groupby("week")[["pred", "actual"]].apply(lambda t: pd.Series({
        "acc": (t.pred == t.actual).mean(), "neu": (t.actual == "neutral").mean(),
        "pneu": (t.pred == "neutral").mean(), "n": len(t)})).reset_index()
    wk = wk[wk.n >= 30]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
    x1 = np.arange(len(lw))
    a1.plot(x1, lw.acc * 100, color=BLUE, lw=1.8, marker="o", ms=4, label="Accuracy")
    a1.plot(x1, lw.bear * 100, color=RED, lw=1.2, ls="--", marker="s", ms=3.5, label="Outcomes that were declines")
    a1.plot(x1, lw.pbull * 100, color=ORANGE, lw=1.2, ls=":", marker="^", ms=3.5, label="Model called a rise")
    a1.set_xticks(x1, [str(w)[5:] for w in lw.week], fontsize=7.5)
    a1.set_ylim(0, 100)
    a1.set_ylabel("%")
    a1.set_title("Live champion (stored, to 17 Sep)", loc="left", fontsize=8.8, weight="bold")
    a1.legend(fontsize=6.6, frameon=False, loc="upper left")
    _style(a1)
    x2 = np.arange(len(wk))
    a2.plot(x2, wk.neu * 100, color=BLUE_T, lw=9, solid_capstyle="round", label="Outcomes 'no significant move'")
    a2.plot(x2, wk.acc * 100, color=BLUE, lw=1.8, marker="o", ms=4, label="Accuracy")
    a2.set_xticks(x2, [str(w)[5:] for w in wk.week], fontsize=7.5)
    a2.set_ylim(0, 100)
    a2.set_title("Calibrated walk-forward model", loc="left", fontsize=8.8, weight="bold")
    a2.legend(fontsize=6.6, frameon=False, loc="lower left")
    _style(a2)
    fig.tight_layout()
    w37 = lw.set_index("week").loc["2026-W37"]
    facts = {"live_weeks": lw.round(3).to_dict(orient="records"),
             "live_corr_acc_bear": float(np.corrcoef(lw.acc, lw.bear)[0, 1]),
             "live_W37": {k: float(w37[k]) for k in ("acc", "bear", "pbull", "n")},
             "oof_pred_neutral_share": float(wk.pneu.mean()),
             "oof_corr_acc_neutral_share": float(np.corrcoef(wk.acc, wk.neu)[0, 1]),
             "oof_weeks": int(len(wk))}
    return b.save(fig, "f5_weekly.png", ["outputs/research/R10_suppression_replay.csv",
                                         "outputs/research/walk_forward_predictions.csv"], facts)


def fig_one_model(b: Builder):
    r13 = b.read("R13_champion_consistency.csv")
    models = ["LogReg", "RandomF", "XGBoost"]
    win = str(r13.pooled_winner.iloc[0])
    gap = (r13[[f"{m}_macro_f1" for m in models]].max(axis=1)
           - r13[f"{win}_macro_f1"]).sort_values().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    yy = np.arange(len(gap))
    ax.hlines(yy, 0, gap, color=BLUE, lw=1.3)
    ax.plot(gap, yy, "o", color=BLUE, ms=4.4)
    ax.axvline(0.02, color=GREY, ls="--", lw=0.9)
    ax.text(0.0215, 4.0, f"{int((gap <= 0.02).sum())} of {len(gap)} stocks are within 0.02;\n"
            f"{int((gap <= 1e-9).sum())} have no shortfall at all", fontsize=7.2, va="center")
    ax.set_yticks([])
    ax.set_ylabel("15 stocks, sorted")
    ax.set_xlabel(f"Macro-F1 shortfall of one pooled model ({win}) against the best model for each stock")
    ax.set_xlim(-0.003, 0.065)
    _style(ax, "x")
    facts = {"winner": win, "leads": int(r13.winner_leads_f1.sum()),
             "within_002": int((gap <= 0.02).sum()), "zero": int((gap <= 1e-9).sum()),
             "mean_gap": float(gap.mean()), "max_gap": float(gap.max())}
    return b.save(fig, "f6_one_model.png", ["outputs/research/R13_champion_consistency.csv"], facts)


def fig_price_history(b: Builder):
    r6 = b.read("R6_long_history.csv")
    pre = r6[r6.period.str.startswith("2019")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    sigs, hs, w = ["REV1", "REV5", "MOM60"], [1, 3, 5], 0.26
    for i, (sgl, c) in enumerate(zip(sigs, [BLUE, ORANGE, AQUA])):
        vals = [float(pre[(pre.signal == sgl) & (pre.h == h)].IC.iloc[0]) for h in hs]
        ts = [float(pre[(pre.signal == sgl) & (pre.h == h)].IC_t.iloc[0]) for h in hs]
        bb = ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=c, label=sgl)
        for bar, v, t in zip(bb, vals, ts):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"t {t:.1f}", ha="center", fontsize=6.6)
    ax.set_xticks(np.arange(3), [f"{h}-day horizon" for h in hs])
    ax.set_ylabel("Mean daily rank IC")
    ax.set_ylim(0, 0.085)
    _style(ax)
    ax.legend(fontsize=7.4, frameon=False, ncol=3, loc="upper right")
    rev1 = pre[(pre.signal == "REV1") & (pre.h == 1)].iloc[0]
    facts = {"REV1_h1_IC": float(rev1.IC), "REV1_h1_t": float(rev1.IC_t),
             "REV1_years": str(rev1.yrs_IC_gt0) if "yrs_IC_gt0" in rev1 else str(rev1.get("yrs_IC>0"))}
    return b.save(fig, "f7_price_history.png", ["outputs/research/R6_long_history.csv"], facts)


# ═══════════════════════════════════════════════════════════════════════════
# F8-F9  outlook record (R14)
# ═══════════════════════════════════════════════════════════════════════════

def fig_outlook_record(b: Builder):
    r = b.read("R14_outlook_reliability.csv")
    rf = r[(r.model == "RandomF")].iloc[0]
    vol = r[(r.model == "volatility only") & r.statement.str.startswith("S2")].iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    groups = ["First half", "Second half", "All 31 dates"]
    q = [rf.quiet_move_rate_first_half, rf.quiet_move_rate_second_half, rf.quiet_move_rate]
    o = [rf.rest_move_rate_first_half, rf.rest_move_rate_second_half, rf.rest_move_rate]
    xx, w = np.arange(3), 0.36
    b1 = a1.bar(xx - w / 2 - 0.01, np.array(q) * 100, w, color=BLUE, label="Called quieter than most (5 a day)")
    b2 = a1.bar(xx + w / 2 + 0.01, np.array(o) * 100, w, color=LIGHT, label="Other 10 stocks")
    for bar in list(b1) + list(b2):
        a1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.0f}%",
                ha="center", fontsize=7)
    a1.set_xticks(xx, groups)
    a1.set_ylim(0, 72)
    a1.set_ylabel("% with a move above 1%")
    a1.set_title("Quieter-than-most call, RandomForest", loc="left", fontsize=8.8, weight="bold")
    a1.legend(fontsize=6.8, frameon=False, loc="upper left")
    _style(a1)

    labels = ["Model's quiet five", "Volatility's quiet five", "Both agree", "Other ten\n(model rule)"]
    vals = [vol.model_quiet_move_rate_same_rows, vol.quiet_move_rate, vol.both_quiet_move_rate,
            vol.model_rest_move_rate_same_rows]
    cols = [BLUE, ORANGE, AQUA, LIGHT]
    yy = np.arange(4)[::-1]
    a2.barh(yy, np.array(vals) * 100, color=cols, height=0.6)
    for y, v in zip(yy, vals):
        a2.text(v * 100 + 1, y, f"{v * 100:.0f}%", va="center", fontsize=7)
    a2.set_yticks(yy, labels, fontsize=7.2)
    a2.set_xlim(0, 70)
    a2.set_xlabel("% with a three-session move above 1%")
    a2.set_title("Same 420 rows: model vs volatility", loc="left", fontsize=8.8, weight="bold")
    _style(a2, "x")
    fig.tight_layout()
    facts = {k: float(rf[k]) for k in ("quiet_moves", "quiet_n", "quiet_move_rate", "rest_moves",
                                       "rest_n", "rest_move_rate",
                                       "quiet_move_rate_first_half", "rest_move_rate_first_half",
                                       "quiet_move_rate_second_half", "rest_move_rate_second_half",
                                       "dates_quiet_strictly_lower", "dates_tied", "p_dates_one_sided",
                                       "nonoverlap_quiet_move_rate", "nonoverlap_rest_move_rate",
                                       "nonoverlap_dates_quiet_strictly_lower", "p_nonoverlap_one_sided",
                                       "auc_move", "auc_direction_among_moves", "lean_sign_hit")}
    facts["event_definition"] = str(rf.event_definition)
    facts.update({"vol_quiet_move_rate": float(vol.quiet_move_rate),
                  "vol_rest_move_rate": float(vol.rest_move_rate),
                  "model_quiet_same_rows": float(vol.model_quiet_move_rate_same_rows),
                  "model_rest_same_rows": float(vol.model_rest_move_rate_same_rows),
                  "both_quiet_move_rate": float(vol.both_quiet_move_rate),
                  "both_quiet_n": int(vol.both_quiet_n), "auc_model_same_rows": float(vol.auc_move_model_same_rows),
                  "auc_vol": float(vol.auc_move)})
    return b.save(fig, "f8_outlook_record.png",
                  ["outputs/research/R14_outlook_reliability.csv"], facts)


def fig_move_calibration(b: Builder):
    c = b.read("R14_move_calibration.csv")
    r = b.read("R14_outlook_reliability.csv")
    rng = r[r.statement.str.startswith("S1")].iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
    a1.plot([0.3, 0.7], [0.3, 0.7], color=GREY, ls="--", lw=0.9)
    sz = 18 + 1.4 * c.n
    a1.scatter(c.stated_mean, c.observed_move_rate, s=sz, color=BLUE, alpha=0.85, zorder=3)
    for _, row in c.iterrows():
        a1.text(row.stated_mean + 0.008, row.observed_move_rate - 0.035, f"n={int(row.n)}",
                fontsize=6.6, color=INK2)
    a1.set_xlim(0.33, 0.68)
    a1.set_ylim(-0.03, 0.75)
    a1.set_xlabel("Stated 1 − P(neutral)")
    a1.set_ylabel("Observed share outside the label band")
    a1.set_title("Calibration on the model's own label band", loc="left", fontsize=8.8, weight="bold")
    _style(a1)

    lab = ["First half", "Second half", "All"]
    vals = [rng.range_coverage_first_half, rng.range_coverage_second_half, rng.range_coverage]
    a2.bar(np.arange(3), np.array(vals) * 100, color=[AQUA, AQUA, BLUE], width=0.55)
    for i, v in enumerate(vals):
        a2.text(i, v * 100 + 1.5, f"{v * 100:.0f}%", ha="center", fontsize=7.2)
    a2.axhline(68.3, color=GREY, ls="--", lw=0.9)
    a2.text(-0.45, 97, "dashed: 68% expected if returns were normal", ha="left", va="top",
            fontsize=6.4, color=INK2)
    a2.set_xticks(np.arange(3), lab)
    a2.set_ylim(0, 100)
    a2.set_xlim(-0.5, 2.5)
    a2.set_ylabel("% inside ±s·√3")
    a2.set_title("Typical range held", loc="left", fontsize=8.8, weight="bold")
    _style(a2)
    fig.tight_layout()
    facts = {"bins": c.round(3).to_dict(orient="records"),
             "range_coverage": float(rng.range_coverage), "range_n": int(rng.n)}
    return b.save(fig, "f9_move_calibration.png",
                  ["outputs/research/R14_move_calibration.csv",
                   "outputs/research/R14_outlook_reliability.csv"], facts)


ALL = [fig_architecture, fig_net_score, fig_rule_tilt, fig_calibration, fig_weekly,
       fig_one_model, fig_price_history, fig_outlook_record, fig_move_calibration]


def build(root, out=None):
    b = Builder(root, out)
    paths = [str(f(b)) for f in ALL]
    (b.out / "sources.json").write_text(json.dumps(b.sources, indent=1, default=float))
    return paths
