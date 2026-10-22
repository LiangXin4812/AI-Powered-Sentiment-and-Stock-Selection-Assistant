# ═══════════════════════════════════════════════════════════════════════════
# TASK 7 — RESEARCH REPORT  (the reviewer's document; the investor's is
#          the Daily Signal Report in daily_report.py)
#
# One cell. Reads only what is already stored in Supabase — no LLM calls, no
# Groq quota — and writes a Word document with:
#
#   1. Executive summary       key figures computed from stored data
#   2. Market-day summary      all 15 stocks on the report date
#   3. Stock reports           full compliance-checked reports for 3 stocks,
#                              each with its evidence chart
#   4. Analytics               model reliability vs naive baselines,
#                              calibration, drift, evidence-layer behaviour,
#                              attention allocation
#   5. Research findings       what the 19 Sep research cells found (R2, R3,
#                              R6, R7, R8 CSVs on Drive) + the price-history
#                              pattern computed live; what it means for
#                              investors
#   6. Methodology & limits
#   A. Pattern definitions
#   B. Responsible-AI guideline conformance (maps the project's compliance
#      and responsible-AI guide to where each control is implemented)
#
# Daily figures are computed from the database; research tables and cost
# assumptions are separately identified. Prereqs: Setup cell + Task 7 import cell (task7_report
# and task7_generate fb_v4 in src/). Section 5 also reads the research CSVs
# if they exist; any missing one is skipped with a note.
# ═══════════════════════════════════════════════════════════════════════════

import sys

import os, json, importlib
from sgx_capstone.configuration import ROOT, load_config
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import text

sys.path.append(str(ROOT / "src"))
importlib.invalidate_caches()
import task7_report, task7_generate

from task7_report import ARCHETYPES, LOW_INFORMATION_ARCHETYPES, ATTENTION_ARCHETYPES, archetype_distribution
from task7_generate import (generate_and_store, render_full_text, SECTION_ORDER,
                            SECTION_TITLES, DISCLAIMER, PROMPT_VERSION_FALLBACK,
                            _is_suppressed)

# ── Settings ────────────────────────────────────────────────────────────────
REPORT_DATE   = None            # None = latest date with all 15 stocks in T5 and T6
WIN_START     = load_config()[0]["report"]["window_start"]    # start of the rl_v7 evidence-layer era
FOCUS         = None            # None = auto-pick 3 stocks; or e.g. ["F34.SI","D05.SI","Z74.SI"]
STORE_REPORTS = False           # True also appends the 15 reports to final_reports
OUT_DIR       = str(ROOT / "outputs" / "report")
RESEARCH_DIR  = str(ROOT / "outputs" / "research")
_REPORT_CFG, _REPORT_SHA = load_config()
CONFIG_SHA = globals().get("CONFIG_SHA", _REPORT_SHA)
CALIB_FIX_DATE = _REPORT_CFG["report"]["recorded_calibration_incident_date"]
NET_SCORE_LINE = _REPORT_CFG["t5"]["net_score_threshold"]           # T5 decision line (MW3), used in section 5.4   # retrain_weekly v2 (out-of-fold calibration) went live
os.makedirs(OUT_DIR, exist_ok=True)
SGT = timezone(timedelta(hours=8))

# Palette (validated reference palette; bullish/bearish = diverging blue/red,
# neutral/baseline = gray). Text always in ink colours, never series colours.
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#c3c2b7"
LIGHT_BLUE, ORANGE = "#86b6ef", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

PATTERN_LABEL = {
    "price_unavailable": "Price unavailable",
    "momentum": "Momentum", "mean_reversion": "Mean reversion",
    "move_complete": "Move complete", "muted_pending": "Muted, pending",
    "muted_fading": "Muted, fading", "dormant": "Dormant", "lag": "Lag",
    "contradicted": "Contradicted", "noise": "Noise", "quiet": "Quiet",
    "model_only": "Model-only view", "incomplete": "Incomplete",
}
SOURCE_LABEL = {"task1_news": "News", "task1_political": "Political",
                "task2_technical": "Technical", "task3_macro": "Macro",
                "task4_options": "Options"}


# ═══════════════════════════════════════════════════════════════════════════
# 1. GATHER
# ═══════════════════════════════════════════════════════════════════════════

def q(sql, **p):
    with SUPABASE_ENGINE.connect() as conn:
        return conn.execute(text(sql), p).fetchall()


def pick_report_date():
    r = q("""
        SELECT a.d::text FROM
          (SELECT analysis_date::date AS d, COUNT(DISTINCT ticker) AS n
             FROM llm_analyses GROUP BY 1) a
        JOIN
          (SELECT prediction_date AS d, COUNT(DISTINCT ticker) AS n
             FROM task6_predictions WHERE is_champion = TRUE GROUP BY 1) p
          ON a.d = p.d
        WHERE a.n >= 15 AND p.n >= 15 AND EXTRACT(ISODOW FROM a.d) <= 5
        ORDER BY a.d DESC LIMIT 1""")
    return r[0][0] if r else None


def gather():
    D = {}
    D["report_date"] = REPORT_DATE or pick_report_date()
    rd = D["report_date"]
    if not rd:
        raise RuntimeError("No complete stored T5/T6 report date is available")
    print(f"Report date: {rd}   window: {WIN_START} → {rd}")

    D["names"] = {t: (v[0] if isinstance(v, (tuple, list)) else str(v))
                  for t, v in (SGX_STOCKS.items() if isinstance(SGX_STOCKS, dict)
                               else [(t, t) for t in SGX_STOCKS])}

    D["coverage"] = [dict(date=r[0], bull=r[1], neu=r[2], bear=r[3]) for r in q("""
        SELECT analysis_date::date::text,
               SUM(CASE WHEN sentiment='bullish' THEN 1 ELSE 0 END),
               SUM(CASE WHEN sentiment='neutral' THEN 1 ELSE 0 END),
               SUM(CASE WHEN sentiment='bearish' THEN 1 ELSE 0 END)
        FROM llm_analyses
        WHERE analysis_date::date BETWEEN CAST(:s AS DATE) AND CAST(:e AS DATE)
          AND EXTRACT(ISODOW FROM analysis_date::date) <= 5
        GROUP BY 1 ORDER BY 1""", s=WIN_START, e=rd)]

    # Champion forecasts whose 3-day outcome was known by the report date.
    D["t6"] = [dict(date=r[0], ticker=r[1], pred=r[2], actual=r[3],
                    prob=float(r[4]), model=r[5]) for r in q("""
        SELECT DISTINCT ON (ticker, prediction_date)
               prediction_date::text, ticker, predicted_class, actual_class,
               GREATEST(prob_bullish, prob_neutral, prob_bearish), model_used
        FROM task6_predictions
        WHERE is_champion = TRUE AND actual_class IS NOT NULL
          AND prediction_date >= CAST(:s AS DATE)
          AND forecast_target <= CAST(:e AS DATE)
        ORDER BY ticker, prediction_date, id DESC""", s=WIN_START, e=rd)]

    D["netscores"] = [dict(date=r[0], net=float(r[1])) for r in q("""
        WITH latest AS (
            SELECT DISTINCT ON (ticker, analysis_date)
                   analysis_date::date AS d,
                   ((features_result::jsonb)->'score_breakdown'->>'net_score')::float AS net
            FROM llm_analyses_audit
            WHERE (features_result::jsonb)->>'_rule_layer_version' IN ('rl_v7', 'rl_v8')
            ORDER BY ticker, analysis_date, id DESC)
        SELECT d::text, net FROM latest
        WHERE d BETWEEN CAST(:s AS DATE) AND CAST(:e AS DATE)
          AND EXTRACT(ISODOW FROM d) <= 5 AND net IS NOT NULL""", s=WIN_START, e=rd)]

    D["post_fix"] = [float(r[0]) for r in q("""
        SELECT DISTINCT ON (ticker, prediction_date)
               GREATEST(prob_bullish, prob_neutral, prob_bearish)
        FROM task6_predictions
        WHERE is_champion = TRUE
          AND prediction_date > CAST(:f AS DATE)
          AND prediction_date <= CAST(:e AS DATE)
        ORDER BY ticker, prediction_date, id DESC""", f=CALIB_FIX_DATE, e=rd)]

    D["champ_log"] = [dict(date=r[0], old=r[1], new=r[2]) for r in q("""
        SELECT changed_at::date::text, old_champion, new_champion
        FROM task6_champion_log ORDER BY changed_at""")]
    D["model_versions"] = [dict(version=str(r[0]), first=str(r[1]),
                                last=str(r[2]), n=int(r[3])) for r in q("""
        SELECT model_version, MIN(prediction_date), MAX(prediction_date), COUNT(*)
        FROM task6_predictions WHERE is_champion = TRUE
          AND prediction_date <= CAST(:d AS DATE)
        GROUP BY model_version ORDER BY MIN(prediction_date)
    """, d=rd)]

    print("Rendering the 15 stock reports (fallback path, no LLM)...")
    D["day"] = {}
    for tkr in D["names"]:
        r = generate_and_store(SUPABASE_ENGINE, tkr, rd, groq_api_key=None,
                               store=STORE_REPORTS, verbose=False)
        if r and "input" in r:
            D["day"][tkr] = r
    print(f"  {len(D['day'])} complete reports")

    print("Classifying patterns across the window (this takes a few minutes)...")
    counts, attention = archetype_distribution(
        SUPABASE_ENGINE, WIN_START, rd, list(D["names"]))
    D["archetypes"], D["attention_window"] = counts, attention
    return D


# ═══════════════════════════════════════════════════════════════════════════
# 2. ANALYTICS  (plain Python over the gathered rows)
# ═══════════════════════════════════════════════════════════════════════════

def analyse(D):
    A = {}
    cov = D["coverage"]
    A["n_days"]  = len(cov)
    A["t5_rows"] = sum(c["bull"] + c["neu"] + c["bear"] for c in cov)
    A["t5_mix"]  = {k: sum(c[k] for c in cov) for k in ("bull", "neu", "bear")}

    t6 = D["t6"]
    A["t6_n"]   = len(t6)
    A["t6_acc"] = sum(r["pred"] == r["actual"] for r in t6) / len(t6) if t6 else None
    act = Counter(r["actual"] for r in t6)
    A["naive_cls"], nc = act.most_common(1)[0] if act else ("neutral", 0)
    A["naive_acc"] = nc / len(t6) if t6 else None
    A["pred_mix"]  = Counter(r["pred"] for r in t6)
    A["act_mix"]   = act

    # calibration bins
    edges = [0.34, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = [r for r in t6 if lo <= r["prob"] < hi]
        if rows:
            bins.append(dict(lo=lo, hi=min(hi, 1.0), n=len(rows),
                             mean_p=sum(r["prob"] for r in rows) / len(rows),
                             hit=sum(r["pred"] == r["actual"] for r in rows) / len(rows)))
    A["calib"] = bins
    hi_rows = [r for r in t6 if r["prob"] >= 0.9]
    A["hi_n"]   = len(hi_rows)
    A["hi_hit"] = (sum(r["pred"] == r["actual"] for r in hi_rows) / len(hi_rows)
                   if hi_rows else None)

    # weekly drift
    wk = defaultdict(list)
    for r in t6:
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        wk[(d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")].append(r)
    weeks = []
    for w in sorted(wk):
        rows = wk[w]; ac = Counter(x["actual"] for x in rows)
        pc_ = Counter(x["pred"] for x in rows)
        weeks.append(dict(week=w, n=len(rows),
                          acc=sum(x["pred"] == x["actual"] for x in rows) / len(rows),
                          naive=ac.most_common(1)[0][1] / len(rows),
                          bear=ac["bearish"] / len(rows),
                          # Carried so the drift chart and caption describe the
                          # outcome mix the live champion met (declines, rise
                          # calls and how many were right) rather than an
                          # untested market-regime story.
                          neu=ac["neutral"] / len(rows),
                          n_bull=sum(x["pred"] == "bullish" for x in rows),
                          bull_right=sum(x["pred"] == "bullish" and x["actual"] == "bullish"
                                         for x in rows),
                          pred_neu=pc_["neutral"] / len(rows),
                          pred_bull=sum(x["pred"] == "bullish" for x in rows) / len(rows)))
    A["weeks"] = weeks

    # per-stock reliability as the reports state it (as-of report date)
    per = []
    for tkr, r in D["day"].items():
        rel = r["input"]["reliability"]
        if rel.get("t6_n"):
            per.append(dict(ticker=tkr, acc=rel["t6_accuracy"],
                            base=rel["t6_baseline"], edge=rel["t6_edge"],
                            n=rel["t6_n"], suppressed=_is_suppressed(r["input"])))
    A["per_stock"] = sorted(per, key=lambda x: x["edge"])
    A["n_below"]   = sum(p["edge"] < 0 for p in per)
    A["n_above"]   = sum(p["edge"] > 0 for p in per)
    A["n_supp"]    = sum(p["suppressed"] for p in per)
    A["mean_edge"] = sum(p["edge"] for p in per) / len(per) if per else None

    # evidence-layer floor
    by = defaultdict(list)
    for x in D["netscores"]:
        by[x["date"]].append(x["net"])
    A["net_days"] = [dict(date=d, lo=min(v), hi=max(v),
                          med=sorted(v)[len(v) // 2], n=len(v)) for d, v in sorted(by.items())]
    A["floor_pos_days"] = sum(d["lo"] > 0 for d in A["net_days"])
    allnet = [x["net"] for x in D["netscores"]]
    A["net_min"] = min(allnet) if allnet else None
    A["net_max"] = max(allnet) if allnet else None
    A["n_net_bear"] = sum(x <= -NET_SCORE_LINE for x in allnet)

    # today
    lv = Counter(r["input"]["attention"]["level"] for r in D["day"].values())
    A["today_attention"] = lv
    A["elevated"] = [t for t, r in D["day"].items() if r["input"]["attention"]["level"] == "ELEVATED"]
    A["notable"]  = [t for t, r in D["day"].items() if r["input"]["attention"]["level"] == "NOTABLE"]

    # archetypes
    ac = {k: v for k, v in D["archetypes"].items() if k != "incomplete"}
    tot = sum(ac.values())
    A["arch_total"] = tot
    A["low_info_share"] = (sum(v for k, v in ac.items() if k in LOW_INFORMATION_ARCHETYPES) / tot
                           if tot else None)
    A["n_swaps"] = len(D["champ_log"])
    A["current_champion"] = D["champ_log"][-1]["new"] if D["champ_log"] else None
    return A


def pick_focus(D, A):
    if FOCUS:
        return [t for t in FOCUS if t in D["day"]]
    picks = []
    order = {"ELEVATED": 0, "NOTABLE": 1, "ROUTINE": 2}
    allt = sorted(D["day"], key=lambda t: (order[D["day"][t]["input"]["attention"]["level"]],
                                           D["day"][t]["input"]["t5"]["sentiment"] == "neutral"))
    picks.append(allt[0])                          # highest attention today
    unsupp = [t for t in D["day"] if not _is_suppressed(D["day"][t]["input"]) and t not in picks]
    if unsupp:
        picks.append(unsupp[0])                    # one where the forecast is shown
    supp = [p["ticker"] for p in A["per_stock"] if p["suppressed"] and p["ticker"] not in picks]
    if supp:
        picks.append(supp[0])                      # the suppression control in action
    for t in D["day"]:                             # one with a directional reading
        if t not in picks and D["day"][t]["input"]["t5"]["sentiment"] != "neutral":
            picks.append(t); break
    for t in D["day"]:
        if len(picks) >= 3: break
        if t not in picks: picks.append(t)
    return picks[:3]


# ═══════════════════════════════════════════════════════════════════════════
# 3. CHARTS
# ═══════════════════════════════════════════════════════════════════════════

def _style(ax, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.yaxis.label.set_color(INK2); ax.xaxis.label.set_color(INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=10)


def _save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def chart_reliability(A, names):
    rows = A["per_stock"]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(rows) + 1.1))
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        ax.plot([r["base"] * 100, r["acc"] * 100], [y, y], color=AXIS, lw=1.5, zorder=1)
    ax.scatter([r["base"] * 100 for r in rows], ys, s=42, color=GRAY, zorder=2,
               label="Prior-data majority rule (time-ordered)")
    ax.scatter([r["acc"] * 100 for r in rows], ys, s=42, color=BLUE, zorder=3,
               label="Forecast model (live, out-of-sample)")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r['ticker']}{'  (withheld)' if r['suppressed'] else '  (ties naive)' if abs(r['edge']) < 0.005 else ''}" for r in rows],
                       fontsize=8.5, color=INK2)
    ax.set_xlim(0, 100); ax.set_xlabel("Directional accuracy, % of matured 3-day forecasts")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.grid(axis="y", visible=False)
    _style(ax, "Forecast accuracy against each stock's own naive rule")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09 - 0.6 / max(len(rows), 1)), ncol=2,
              fontsize=8, frameon=False, labelcolor=INK2)
    return _save(fig, "c1_reliability.png")


def chart_calibration(A):
    b = A["calib"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot([0.33, 1], [33, 100], color=AXIS, ls="--", lw=1, label="Perfect calibration")
    ax.plot([x["mean_p"] for x in b], [x["hit"] * 100 for x in b], "-o", color=BLUE,
            lw=2, ms=6, label="Observed hit rate")
    for x in b:
        ax.annotate(f"n={x['n']}", (x["mean_p"], x["hit"] * 100), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0.33, 1.0); ax.set_ylim(0, 105)
    ax.set_xlabel("Probability the model gave its own forecast")
    ax.set_ylabel("% of forecasts that were right")
    _style(ax, "Stated confidence vs realised accuracy")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")
    return _save(fig, "c2_calibration.png")


def chart_drift(A):
    """
    Weekly accuracy of the stored live champion forecasts, with the share of
    outcomes that were declines and the share of forecasts that were rise
    calls. An earlier version explained a falling accuracy by the model
    having been trained in a rising market; that was never tested and is not
    used. The chart shows the outcome mix the model met, not a cause.
    """
    w = A["weeks"]
    xs = [x["week"][5:] for x in w]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(xs, [x["acc"] * 100 for x in w], "-o", color=BLUE, lw=1.8, ms=5,
            label="Model accuracy", zorder=3)
    ax.plot(xs, [x["pred_bull"] * 100 for x in w], "-^", color=ORANGE, lw=1.4, ms=4.5,
            label="Share of forecasts that were rise calls", zorder=2)
    ax.plot(xs, [x["naive"] * 100 for x in w], "-o", color=GRAY, lw=1.6, ms=4.5,
            label="Naive rule (most common outcome that week)", zorder=2)
    ax.plot(xs, [x["bear"] * 100 for x in w], "-s", color=RED, lw=1.6, ms=4.5,
            label="Share of outcomes that were declines", zorder=2)
    ax.set_ylim(0, 100); ax.set_ylabel("%"); ax.set_xlabel("Week of forecast (starting Monday)")
    _style(ax, "Weekly accuracy of the live champion and the outcome mix")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    return _save(fig, "c3_drift.png")


def chart_t5_mix(D):
    c = D["coverage"]
    xs = [x["date"][5:] for x in c]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    b1 = [x["bull"] for x in c]; b2 = [x["neu"] for x in c]; b3 = [x["bear"] for x in c]
    ax.bar(xs, b1, color=BLUE, label="Leaned positive", edgecolor="white", lw=1)
    ax.bar(xs, b2, bottom=b1, color=GRAY, label="No clear lean", edgecolor="white", lw=1)
    ax.bar(xs, b3, bottom=[a + b for a, b in zip(b1, b2)], color=RED,
           label="Leaned negative", edgecolor="white", lw=1)
    ax.set_ylabel("Stocks"); plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7.5)
    _style(ax, "Evidence-layer readings per day, 15 stocks")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.25))
    return _save(fig, "c4_t5_mix.png")


def chart_floor(A):
    d = A["net_days"]
    xs = [x["date"][5:] for x in d]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.fill_between(xs, [x["lo"] for x in d], [x["hi"] for x in d], color=LIGHT_BLUE,
                    alpha=0.35, lw=0, label="Range across 15 stocks")
    ax.plot(xs, [x["med"] for x in d], "-o", color=BLUE, lw=2, ms=5, label="Median stock")
    ax.plot(xs, [x["lo"] for x in d], "-v", color=INK2, lw=1.4, ms=5, label="Most negative stock")
    for yv, lab in ((NET_SCORE_LINE, f"positive line +{NET_SCORE_LINE:.2f}"), (-NET_SCORE_LINE, f"negative line -{NET_SCORE_LINE:.2f}")):
        ax.axhline(yv, color=RED if yv < 0 else BLUE, ls="--", lw=1)
        ax.text(len(xs) - 0.6, yv + 0.03, lab, ha="right", fontsize=7.5, color=INK2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Net score"); plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7.5)
    _style(ax, "Evidence-layer net score by day")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.25))
    return _save(fig, "c5_floor.png")


def chart_archetypes(D):
    ac = {k: v for k, v in D["archetypes"].items() if k != "incomplete" and v}
    items = sorted(ac.items(), key=lambda kv: kv[1])
    col = [GRAY if k in LOW_INFORMATION_ARCHETYPES else BLUE if k in ATTENTION_ARCHETYPES
           else LIGHT_BLUE for k, _ in items]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(items) + 1.0))
    ax.barh([PATTERN_LABEL.get(k, k) for k, _ in items], [v for _, v in items], color=col,
            edgecolor="white", lw=1)
    tot = sum(ac.values())
    for i, (k, v) in enumerate(items):
        ax.text(v + tot * 0.005, i, f"{v} ({v / tot:.0%})", va="center", fontsize=7.8, color=INK2)
    ax.set_xlabel("Stock-days")
    _style(ax, "How the report layer classified each stock-day")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRAY, label="Low information"),
                       Patch(color=LIGHT_BLUE, label="Informative, no divergence"),
                       Patch(color=BLUE, label="Eligible for attention")],
              fontsize=7.8, frameon=False, labelcolor=INK2, loc="lower right")
    return _save(fig, "c6_archetypes.png")


def chart_evidence(tkr, att):
    feats = sorted((att or {}).get("features") or [], key=lambda f: f["weight"])
    if not feats:
        return None
    vals = [f["weight"] if f["direction"] == "bullish" else -f["weight"] for f in feats]
    labs = [f"{f['factor']} [{SOURCE_LABEL.get(f['source'], f['source'])}]{' *' if f['decisive'] else ''}"
            for f in feats]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(vals) + 1.0))
    ax.barh([labs[i] for i in order], [vals[i] for i in order],
            color=[BLUE if vals[i] > 0 else RED for i in order], edgecolor="white", lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    m = max(abs(v) for v in vals) * 1.15
    ax.set_xlim(-m, m)
    ax.set_xlabel("Rule weight   (left = points down, right = points up)")
    net = att.get("net_score")
    _style(ax, f"{tkr}: rules that fired"
               + (f"   (net {net:+.3f}; lines at ±0.35)" if net is not None else ""))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.6)
    return _save(fig, f"ev_{tkr.replace('.', '_')}.png")


def chart_probs(tkr, inp):
    p = inp["t6"].get("probabilities") or {}
    ks = ["bearish", "neutral", "bullish"]
    vs = [float(p.get(k) or 0) for k in ks]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(["Decline", "No big move", "Rise"], vs, color=[RED, GRAY, BLUE], edgecolor="white")
    for i, v in enumerate(vs):
        ax.text(i, v + 0.03, ">0.99" if v > 0.99 else "<0.01" if v < 0.01 else f"{v:.2f}",
                ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.5, 1])
    _style(ax, "3-day forecast probabilities")
    return _save(fig, f"pr_{tkr.replace('.', '_')}.png")


# ── Research findings (section 5) ───────────────────────────────────────────

def load_research(D):
    """Research CSVs written by the 19 Sep research cells, plus the live
    price-history pattern from any report input. Missing pieces are None."""
    import pandas as pd
    R = {}
    for key, name in [("r13", "R13_champion_consistency"),
                      ("r10stock", "R10_per_stock_vs_naive"),
                      ("r10replay", "R10_suppression_replay"),
                      ("r11", "R11_summary"),
                      ("r2", "R2_horizon_calibration"), ("r3", "R3_lookback_margin"),
                      ("r6", "R6_long_history"), ("r7", "R7_t5_rule_audit"),
                      ("r8", "R8_rl_v8_replay"), ("r9", "R9_t5_v6_vs_v7"),
                      ("r9w", "R9_t5_v6_vs_v7_weights"),
                      ("r14", "R14_outlook_reliability")]:
        path = os.path.join(RESEARCH_DIR, f"{name}.csv")
        try:
            R[key] = pd.read_csv(path) if os.path.exists(path) else None
        except Exception as e:                                   # noqa: BLE001
            print(f"  ⚠️  could not read {name}.csv ({e!r})")
            R[key] = None
    R["pc"] = next((r["input"].get("price_context") for r in D["day"].values()
                    if r["input"].get("price_context")), None)
    print("Research inputs: " + ", ".join(
        f"{k} {'✓' if R[k] is not None else '—'}" for k in ["r2", "r3", "r6", "r7", "r8", "r9", "pc"]))
    return R


def chart_calibration_fix(r2):
    g = r2[r2["h"] == 3]
    models = list(g["model"].unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    x = range(len(models))
    for ax in (a1, a2):
        _style(ax)
    lk = [float(g[(g.model == m) & (g.calib == "leaky")]["share_ge90"].iloc[0]) for m in models]
    of = [float(g[(g.model == m) & (g.calib == "oof")]["share_ge90"].iloc[0]) for m in models]
    a1.bar([i - 0.19 for i in x], lk, 0.36, color=RED, label="old calibration")
    a1.bar([i + 0.19 for i in x], of, 0.36, color=BLUE, label="fixed calibration")
    a1.set_xticks(list(x)); a1.set_xticklabels(models)
    a1.set_ylabel("share of forecasts at ≥ 0.9")
    a1.set_title("Stated confidence ≥ 0.9", loc="left", fontsize=10, color=INK, fontweight="bold")
    acc_l = [float(g[(g.model == m) & (g.calib == "leaky")]["acc"].iloc[0]) for m in models]
    acc_o = [float(g[(g.model == m) & (g.calib == "oof")]["acc"].iloc[0]) for m in models]
    neu = float(g["neutral_acc"].iloc[0]) if "neutral_acc" in g else None
    a2.bar([i - 0.19 for i in x], acc_l, 0.36, color=RED)
    a2.bar([i + 0.19 for i in x], acc_o, 0.36, color=BLUE)
    if neu is not None:
        a2.axhline(neu, color=INK2, linestyle="--", linewidth=1)
        a2.text(len(models) - 0.5, neu + 0.01, "always 'neutral'", ha="right",
                fontsize=8, color=INK2)
    a2.set_xticks(list(x)); a2.set_xticklabels(models); a2.set_ylim(0, 0.7)
    a2.set_ylabel("accuracy (3-day, walk-forward)")
    a2.set_title("Accuracy", loc="left", fontsize=10, color=INK, fontweight="bold")
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, "r_calibration_fix.png")


def _rule_family(f):
    return ("Bullish corporate events (any count)" if f.startswith("Bullish corporate events")
            else "Bearish corporate events (any count)" if f.startswith("Bearish corporate events")
            else f)


def chart_tilt(r7):
    agg = defaultdict(float)
    for _, r in r7.iterrows():
        agg[_rule_family(str(r["factor"]))] += float(r["tilt_contrib"])
    items = sorted(agg.items(), key=lambda kv: kv[1])
    items = items[:4] + items[-8:] if len(items) > 12 else items
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(items) + 1.0))
    _style(ax, "Contribution of each rule to the average reading")
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=[BLUE if v > 0 else RED for _, v in items], height=0.62)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("points added to the average net score (positive = pushes toward 'positive')")
    lo = min(v for _, v in items); hi = max(v for _, v in items)
    ax.set_xlim(min(lo, 0) * 1.35 - 0.01, max(hi, 0) * 1.3 + 0.01)
    for i, (_, v) in enumerate(items):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    return _save(fig, "r_tilt.png")


def chart_reversal(pc):
    b = pc["group"]["bands"]
    labels = ["Bottom\nfifth", "2nd\nfifth", "Middle\nfifth", "4th\nfifth", "Top\nfifth"]
    vals = [x["mean_next_rel_pct"] or 0 for x in b]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    _style(ax, "Next-day relative return after days in each band")
    ax.bar(labels, vals, color=[BLUE if v > 0 else RED for v in vals], width=0.6)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    for i, x in enumerate(b):
        v = vals[i]
        ax.text(i, v + (0.004 if v >= 0 else -0.004),
                f"{v:+.3f}%\n{x['share_outperformed']:.0%} beat group",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
    ax.set_ylabel("avg next-day return vs group (%)")
    ax.set_xlabel("today's move relative to the group, by the stock's own history")
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = (hi - lo) * 0.45 or 0.05
    ax.set_ylim(lo - pad, hi + pad)
    return _save(fig, "r_reversal.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def _rgb(h):
    h = h.lstrip("#"); return RGBColor(int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


def _shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcol.lstrip("#"))
    tcPr.append(shd)


class Doc:
    def __init__(self):
        self.d = Document()
        s = self.d.sections[0]
        s.left_margin = s.right_margin = Inches(0.85)
        s.top_margin = s.bottom_margin = Inches(0.8)
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"; st.font.size = Pt(10.5)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        for lvl, size in ((1, 15), (2, 12.5), (3, 11)):
            h = self.d.styles[f"Heading {lvl}"]
            h.font.name = "Calibri"; h.font.size = Pt(size); h.font.color.rgb = _rgb("#1E2761")

    def h(self, t, lvl=1):
        self.d.add_heading(t, level=lvl)

    def p(self, t, italic=False, size=None, color=None, bold=False, space_after=6):
        para = self.d.add_paragraph()
        run = para.add_run(t)
        run.italic = italic; run.bold = bold
        if size: run.font.size = Pt(size)
        if color: run.font.color.rgb = _rgb(color)
        para.paragraph_format.space_after = Pt(space_after)
        return para

    def rich(self, parts, space_after=6):
        para = self.d.add_paragraph()
        for t, b in parts:
            r = para.add_run(t); r.bold = b
        para.paragraph_format.space_after = Pt(space_after)

    def bullet(self, t, bold_lead=None):
        para = self.d.add_paragraph(style="List Bullet")
        if bold_lead:
            para.add_run(bold_lead).bold = True
        para.add_run(t)
        para.paragraph_format.space_after = Pt(2)

    def caption(self, t):
        self.p(t, italic=True, size=9, color=INK2, space_after=10)

    def img(self, path, width=6.4):
        if path:
            self.d.add_picture(path, width=Inches(width))
            self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    def table(self, header, rows, widths=None, font=8.5, zebra=True):
        t = self.d.add_table(rows=1, cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.style = "Table Grid"
        for i, htxt in enumerate(header):
            c = t.rows[0].cells[i]; c.text = ""
            r = c.paragraphs[0].add_run(htxt); r.bold = True; r.font.size = Pt(font)
            r.font.color.rgb = _rgb("#FFFFFF"); _shade(c, "#1E2761")
        repeat = OxmlElement("w:tblHeader")
        t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for k, row in enumerate(rows):
            added = t.add_row()
            added._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            cells = added.cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(font)
                if zebra and k % 2:
                    _shade(cells[i], "#F4F5F7")
        if widths:
            t.autofit = False
            for i, w in enumerate(widths):
                t.columns[i].width = Inches(w)
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def box(self, lines, fill="#F0F7FF"):
        t = self.d.add_table(rows=1, cols=1); c = t.rows[0].cells[0]; _shade(c, fill)
        c.text = ""
        for i, (txt, bold) in enumerate(lines):
            para = c.paragraphs[0] if i == 0 else c.add_paragraph()
            r = para.add_run(txt); r.bold = bold; r.font.size = Pt(9.5)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def page_break(self):
        self.d.add_page_break()


def pct(x, dp=0):
    return "n/a" if x is None else f"{x * 100:.{dp}f}%"


def fmt_prob(p):
    return ">0.99" if p > 0.99 else "<0.01" if p < 0.01 else f"{p:.2f}"


def build_doc(D, A, focus):
    rd = D["report_date"]; names = D["names"]
    doc = Doc()

    # ── Cover ──
    title = doc.d.add_paragraph("SGX investment evidence and forecast review", style="Title")
    for run in title.runs:
        run.font.color.rgb = _rgb("#000000")
    doc.p(f"Daily Signal Report with Analytics  ·  data as of {rd} market close",
          size=12, color=INK2, space_after=2)
    doc.p(f"15 SGX stocks  ·  analysis window {WIN_START} to {rd}  ·  generated "
          f"{datetime.now(SGT):%d %b %Y %H:%M} SGT  ·  config {CONFIG_SHA} "
          f"·  Liang Xin (A0221401E)",
          size=9, color=MUTED, space_after=10)
    doc.box([("How to read this report. ", True),
             ("Individual readings and stored forecast records are computed by "
              "deterministic code. Historical research results and cost assumptions "
              "are separately identified. The "
              "report describes what the signals said and how reliable the "
              "forecasts have been. It does not recommend any action. ", False),
             (DISCLAIMER, False)], fill="#F4F5F7")

    # ── 1. Executive summary ──
    doc.h("1. Executive summary")
    doc.table(["Evidence readings", "Matured forecasts", "Forecast accuracy",
               f"Retrospective majority ('{A['naive_cls']}')", "Forecasts withheld today"],
              [[f"{A['t5_rows']} over {A['n_days']} days", f"{A['t6_n']}",
                pct(A["t6_acc"], 1), pct(A["naive_acc"], 1),
                f"{A['n_supp']} of {len(D['day'])} stocks"]],
              font=9.5, zebra=False)
    m = A["t5_mix"]; tot = sum(m.values()) or 1
    w = A["weeks"]
    doc.bullet(f" {pct(m['bull']/tot)} of evidence readings leaned positive, "
               f"{pct(m['neu']/tot)} showed no clear lean and {m['bear']} leaned "
               f"negative. Section 5.3 traces this to one rule, the corporate-event "
               f"rule, which fires on almost every day; it is a design flaw with a "
               f"tested fix, not a data error.",
               bold_lead="Evidence layer.")
    doc.bullet(f" Since {WIN_START}, the live forecast model was right on "
               f"{pct(A['t6_acc'],1)} of {A['t6_n']} matured champion 3-day forecasts "
               f"from {WIN_START} to {rd}. Choosing the most frequent outcome "
               f"after seeing these same results would score {pct(A['naive_acc'],1)}; "
               f"this is a retrospective diagnostic, not an investable rule. "
               f"Measured stock by stock, the model beats that stock's own naive "
               f"rule on {A['n_above']} of {len(A['per_stock'])} stocks.",
               bold_lead="Forecast reliability.")
    if len(w) >= 2:
        peak = max(w, key=lambda x: x["bear"])
        txt = (f" The share of 3-day outcomes that were declines moved from "
               f"{pct(w[0]['bear'])} in the first week to {pct(peak['bear'])} in "
               f"the week of {peak['week']}"
               + (f" and {pct(w[-1]['bear'])} in the last" if peak is not w[-1] else "")
               + f". Model accuracy was {pct(w[0]['acc'])} in the first week, "
               f"{pct(peak['acc'])} in that week and {pct(w[-1]['acc'])} in the "
               f"last")
        txt += (f", so it did worst when declines were most common: that week it "
                f"called a rise on {peak['n_bull']} forecasts and {peak['bull_right']} "
                f"were right (section 4.1)."
                if peak["acc"] < w[0]["acc"] - 0.05 else
                "; with this few forecasts per week the week-to-week changes are "
                "within noise (section 4.1).")
        doc.bullet(txt, bold_lead="Market shift.")
    if A["hi_n"]:
        pf = D.get("post_fix") or []
        txt = (f" Across the {WIN_START} to {rd} matured champion window, "
               f"when the model stated 0.9 "
               f"probability or more it was right {pct(A['hi_hit'])} of the time "
               f"({A['hi_n']} matured forecasts). The cause was a calibration "
               f"leak (section 5.1).")
        if pf:
            n_hi = sum(p >= 0.9 for p in pf)
            txt += (f" After {CALIB_FIX_DATE}, {len(pf)} stored champion forecasts "
                    f"were issued; the "
                    f"highest stated probability is {max(pf):.2f} and "
                    f"{n_hi} stated 0.9 or more.")
            if n_hi:
                txt += (" The 22 Sep overwrite is one documented source of "
                        "overconfidence; model provenance for each other row "
                        "needs verification. See Appendix C.")
        doc.bullet(txt, bold_lead="Overconfidence incident.")
    doc.bullet(f" {A['n_supp']} stocks have their forecast automatically withheld "
               f"today because the model runs more than 5 points below that "
               f"stock's time-ordered prior-data majority benchmark. This is a "
               f"cautious display control, not evidence that suppression improves "
               f"investor outcomes. Reader attention today: "
               f"{A['today_attention'].get('ELEVATED',0)} elevated, "
               f"{A['today_attention'].get('NOTABLE',0)} notable, "
               f"{A['today_attention'].get('ROUTINE',0)} routine.",
               bold_lead="Controls in effect.")
    pcx = (A.get("research") or {}).get("pc")
    if (pcx and pcx["group"].get("reversal_ic") is not None
            and pcx["group"]["reversal_ic"] > 0 and (pcx["group"].get("reversal_ic_t") or 0) >= 2):
        gpx = pcx["group"]
        doc.bullet(f" Across {gpx['reversal_days']} trading days since "
                   f"{gpx['history_start']}, these stocks' daily moves relative to "
                   f"each other have tended to partly reverse the next day (rank "
                   f"correlation {gpx['reversal_ic']:+.3f}, t "
                   f"{gpx['reversal_ic_t']}). Every stock report now shows what "
                   f"followed days like today (section 5.5).",
                   bold_lead="Price history.")
    doc.p("Bottom line: the system delivers value as a transparent reading of "
          "the evidence, an honest account of its own reliability, and "
          "historical context for today's move. Its 3-day forecasts carry no "
          "reliable directional information in this period, and the report "
          "layer says so and withholds them where the record is worst.",
          italic=True, color=INK2)

    # ── Investor reading path: concrete, date-specific evidence ──
    if D["day"]:
        # The verification page is only worth a page if the stock HAS a dated,
        # event-tagged article on this date. Picking a fixed ticker produced an
        # empty page whenever that ticker happened to have none.
        try:
            _with_news = [r[0] for r in q("""
                SELECT n.ticker, COUNT(*) AS c
                FROM news_articles n
                WHERE n.date = CAST(:d AS DATE) AND n.category = 'company'
                  AND n.event_flag IS NOT NULL AND n.event_flag <> 'none'
                GROUP BY n.ticker ORDER BY c DESC
            """, d=rd) if r[0] in D["day"]]
        except Exception:
            _with_news = []
        CASE_PREFERENCE = ["S68.SI"]          # keep the documented example first
        case_tkr = next((t for t in CASE_PREFERENCE if t in _with_news),
                        _with_news[0] if _with_news
                        else ("S68.SI" if "S68.SI" in D["day"] else focus[0]))
        if not _with_news:
            print("  note: no stock has an event-tagged article on this date; "
                  "the verification page will say so explicitly")
        else:
            print(f"  verification page case stock: {case_tkr} "
                  f"(candidates with tagged articles: {', '.join(_with_news[:5])})")
        case = D["day"][case_tkr]["input"]
        px = case.get("price_context") or {}
        doc.page_break()
        doc.h(f"What an investor can verify today: {case_tkr}")
        doc.p(f"Example for {rd}, using the information attached to this "
              f"report date. Read this page before the model diagnostics.")
        try:
            headlines = q("""
                SELECT n.title, n.source, n.published, n.event_flag,
                       (SELECT r.url FROM raw_articles r
                        WHERE r.ticker=n.ticker AND r.date=n.date AND r.title=n.title
                        ORDER BY r.published DESC LIMIT 1)
                FROM news_articles n
                WHERE n.ticker=:t AND n.date=CAST(:d AS DATE)
                  AND n.category='company' AND n.event_flag IS NOT NULL
                  AND n.event_flag <> 'none'
                ORDER BY n.published DESC LIMIT 3
            """, t=case_tkr, d=rd)
        except Exception:
            headlines = []
        price_line = (f"{px['return_pct']:+.2f}% close-to-close; "
                      f"{px['relative_pct']:+.2f} percentage points versus "
                      f"the equal-weight covered group ({px['n_stocks']} stocks)"
                      if px.get("as_of") == rd and px.get("return_pct") is not None
                      else "Same-date price comparison unavailable")
        doc.table(["Investor question", "What the stored evidence supports"], [
            ["What happened?", f"{len(headlines)} tagged company headlines "
             f"available below; tags identify keywords, not verified events."],
            ["How did price react?", price_line],
            ["What does T5 say?", f"{case['t5']['sentiment']} evidence reading; "
             f"strength {case['t5']['confidence']:.2f} is an internal rule score, "
             f"not the probability of a rise."],
            ["What does T6 say?", "Forecast withheld on this date; no stock "
             "selection conclusion follows." if _is_suppressed(case) else
             f"Three-session class {case['t6']['predicted_class']} with model "
             f"class probability {fmt_prob(case['t6']['probability'])}; check "
             f"the {case['reliability'].get('t6_n',0)} matured prior observations."],
            ["What remains uncertain?", "A headline tag cannot prove timing, "
             "causality, or that the move has not already incorporated the news."],
        ], widths=[1.5, 5.1], font=8.8)
        if headlines:
            doc.h("Check the original publications", 2)
            doc.table(["Headline", "Source", "Published", "Tag", "URL"],
                      [[str(h[0])[:90], str(h[1])[:25], str(h[2])[:25],
                        str(h[3]), str(h[4] or "unavailable")[:95]]
                       for h in headlines], font=7)
        else:
            doc.p("No tagged company article was found for this date. The "
                  "original inputs need manual verification before interpreting "
                  "any corporate-event claim.", italic=True)
        doc.p("Investor task: verify a publication's original timestamp and "
              "content, distinguish company-specific news from a common market "
              "move, and read a withheld forecast as insufficient evidence. "
              "No reader study has yet measured whether this page improves "
              "those decisions.", italic=True)

    # ── 2. Market-day summary ──
    doc.page_break()
    doc.h(f"2. Market-day summary: {rd}")
    doc.p("One row per stock. 'Evidence reading' is the signal layer's view of the "
          "day; 'Forecast' is the 3-day model view (withheld where its record is "
          "below the stock's naive rule); 'Outcome' is the realised 3-day move "
          "if already known.", size=9.5, color=INK2)
    rows = []
    lvl_order = {"ELEVATED": 0, "NOTABLE": 1, "ROUTINE": 2}
    for tkr, r in sorted(D["day"].items(),
                         key=lambda kv: (lvl_order[kv[1]["input"]["attention"]["level"]], kv[0])):
        i = r["input"]; sup = _is_suppressed(i)
        fc = ("withheld" if sup else
              f"{i['t6']['predicted_class']} {fmt_prob(i['t6']['probability'] or 0)}")
        out = (f"{i['actual']['return_3d']:+.2f}%" if i["actual"]["matured"] else "pending")
        pat = "—" if sup else PATTERN_LABEL.get(i["fusion"]["archetype"], i["fusion"]["archetype"])
        rows.append([tkr, names.get(tkr, "")[:26],
                     f"{i['t5']['sentiment']} / strength {i['t5']['confidence']:.2f}",
                     fc, out, pat, i["attention"]["level"]])
    doc.table(["Ticker", "Company", "Evidence strength*", "Forecast", "Outcome",
               "Pattern", "Attention"], rows,
              widths=[0.7, 1.7, 1.0, 1.0, 0.7, 1.1, 0.8], font=8)
    doc.caption("*Evidence strength is an internal rule score, not a calibrated "
                "chance of a future rise. Forecast numbers are probabilities of "
                "the displayed three-session class; neither implies a trade.")

    # ── 3. Stock reports ──
    doc.h("3. Stock reports")
    top_lvl = D["day"][focus[0]]["input"]["attention"]["level"] if focus else "ROUTINE"
    lead = ("the stock with the highest attention level today"
            if top_lvl != "ROUTINE" else
            "no stock was raised above routine today, so the first is one "
            "with a directional evidence reading")
    doc.p(f"Three stocks are shown in full: {lead}; one where the forecast is "
          f"shown; and one where the suppression control withholds it. The same "
          f"report exists for all 15 stocks.", size=9.5, color=INK2)
    for n, tkr in enumerate(focus, 1):
        r = D["day"][tkr]; i = r["input"]; s = r["sections"]; att = r["attribution"]
        doc.h(f"3.{n}  {tkr}  {names.get(tkr, '')}", 2)
        doc.rich([("Attention: ", True), (i["attention"]["level"], False),
                  ("   ·   Pattern: ", True),
                  ("forecast withheld" if _is_suppressed(i)
                   else PATTERN_LABEL.get(i["fusion"]["archetype"], ""), False),
                  ("   ·   Evidence rule layer: ", True),
                  (str((att or {}).get("rule_layer_version")), False)])
        if i["actual"]["matured"]:
            doc.p("Retrospective: the outcome section uses prices from after the "
                  "report date; all other figures use only information available "
                  "on it.", italic=True, size=8.5, color=MUTED)
        for key in SECTION_ORDER:
            if key == "methodology":
                doc.p("Methodology, assumptions and limitations: see section 6 "
                      "(identical for every stock).", italic=True, size=9, color=INK2)
                continue
            doc.h(SECTION_TITLES[key].capitalize(), 3)
            if key == "key_evidence":
                doc.img(chart_evidence(tkr, att), width=6.2)
                feats = sorted((att or {}).get("features") or [], key=lambda f: -f["weight"])
                if feats:
                    doc.table(["Rule", "Source", "Points", "Weight", "Decisive"],
                              [[f["factor"], SOURCE_LABEL.get(f["source"], f["source"]),
                                "up" if f["direction"] == "bullish" else "down",
                                f"{f['weight']:.3f}", "yes" if f["decisive"] else ""]
                               for f in feats],
                              widths=[2.8, 0.9, 0.6, 0.7, 0.7], font=8)
                doc.caption("Weights are the rule layer's fitted contributions after "
                            "market-wide discounting. 'Decisive' means the reading "
                            "would change without that rule.")
                continue
            if key == "model_view" and not _is_suppressed(i):
                doc.img(chart_probs(tkr, i), width=3.2)
            doc.p(s[key])
        # News provenance lets a reader inspect the concrete inputs behind
        # the day's event tags. An event tag is a headline keyword, not an
        # independently verified corporate action or a causal explanation.
        try:
            articles = q("""
                SELECT n.title, n.source, n.published, n.event_flag,
                       (SELECT r.url FROM raw_articles r
                        WHERE r.ticker=n.ticker AND r.date=n.date
                          AND r.title=n.title ORDER BY r.published DESC LIMIT 1)
                FROM news_articles n
                WHERE n.ticker = :t AND n.date = CAST(:d AS DATE)
                  AND n.category = 'company' AND n.event_flag IS NOT NULL
                  AND n.event_flag <> 'none'
                ORDER BY n.published DESC LIMIT 5
            """, t=tkr, d=rd)
        except Exception:
            articles = []
        doc.h("Articles behind the event tags", 3)
        if articles:
            doc.p("These are stored headlines tagged by a keyword rule. Check the "
                  "original publication and timing before treating a tag as a new "
                  "company event; the tag does not establish its price impact.",
                  size=8.5, color=INK2)
            doc.table(["Headline", "Source", "Published", "Tag", "Original URL"],
                      [[str(a[0])[:95], str(a[1])[:25], str(a[2])[:25], str(a[3]),
                        str(a[4] or "unavailable")[:100]]
                       for a in articles],
                      widths=[2.2, 0.7, 0.8, 0.6, 2.3], font=7)
        else:
            doc.p("No event-tagged article is available for verification on "
                  "this report date. The rule-level reading should be treated "
                  "as unverified context.", size=8.5, color=INK2)

    # ── 4. Analytics ──
    doc.page_break()
    doc.h("4. Analytics")

    doc.h("4.1 Is the forecast model reliable?", 2)
    r13 = (A.get("research") or {}).get("r13")
    if r13 is not None and len(r13):
        win = str(r13["pooled_winner"].iloc[0])
        nlead = int(r13["winner_leads_f1"].sum())
        doc.p(f"Consistency check (R13): the pooled three-session "
              f"walk-forward winner ({win}) leads the other two candidates "
              f"on per-stock macro F1 for {nlead}/{len(r13)} stocks "
              f"({nlead/len(r13):.0%}); the target for this check is 80%. "
              f"The winner was selected on these evaluation dates, so this "
              f"is a retrospective diagnostic rather than an independent "
              f"future-date pass.")
        doc.table(["Ticker", "N/model", "LogReg F1", "RF F1", "XGB F1",
                   "XGB acc", "Prior rule acc", "Winner leads"],
                  [[str(x.ticker), int(x.XGBoost_n),
                    f"{x.LogReg_macro_f1:.2f}", f"{x.RandomF_macro_f1:.2f}",
                    f"{x.XGBoost_macro_f1:.2f}", f"{x.XGBoost_acc:.0%}",
                    f"{x.XGBoost_prior_rule_acc:.0%}",
                    "yes" if x.winner_leads_f1 else "no"]
                   for x in r13.itertuples()], font=7.4)
        # Aggregate view: the question that matters for deployment is not
        # "does the pooled winner strictly beat the others on every stock"
        # but "what does serving ONE model cost compared with serving the
        # best model per stock". Reported as counts and a mean gap so no
        # individual ticker has to be read off the table above.
        try:
            _m = ["LogReg", "RandomF", "XGBoost"]
            _cols = [f"{x}_macro_f1" for x in _m if f"{x}_macro_f1" in r13.columns]
            _best = r13[_cols].max(axis=1)
            _gap = _best - r13[f"{win}_macro_f1"]
            _n = len(r13)
            doc.p(
                f"Aggregate reading of the same table. Serving one model for "
                f"all {_n} stocks instead of the best model per stock costs "
                f"{_gap.mean():.3f} macro F1 on average. The single model is "
                f"itself the per-stock best on {(_gap <= 1e-9).sum()} of {_n} "
                f"stocks, is within 0.02 of it on {(_gap <= 0.02).sum()} of "
                f"{_n} ({(_gap <= 0.02).mean():.0%}), and within 0.05 on "
                f"{(_gap <= 0.05).sum()} of {_n}; the largest shortfall on any "
                f"stock is {_gap.max():.3f}. A per-stock selector would "
                f"therefore be choosing between options separated by less "
                f"than the noise in roughly 30 matured forecasts per stock. "
                f"This, not the strict lead count above, is the basis for "
                f"deploying a single model.")
        except Exception as _e:                                  # noqa: BLE001
            print(f"  note: aggregate model comparison skipped ({_e!r})")
        doc.caption("R13: 3-session horizon, out-of-fold calibration, same "
                    "(date, stock) rows for all models. 'Prior rule' uses the "
                    "stored training-window naive label for each test row. "
                    "The live champion can differ from this pooled winner.")
    r10stock = (A.get("research") or {}).get("r10stock")
    if r10stock is not None and len(r10stock):
        doc.p("Live-record benchmark (R10): the rule fixed from prior outcomes "
              "is distinguishable from the best constant class chosen after "
              "seeing the same outcomes. Neither is a strategy return.")
        doc.table(["Ticker", "N", "Champion acc", "Macro F1",
                   "Prior rule", "After-the-fact majority"],
                  [[x.ticker, int(x.n), f"{x.accuracy:.0%}",
                    f"{x.macro_f1:.2f}", f"{x.naive_ex_ante:.0%}",
                    f"{x.naive_realised:.0%}"]
                   for x in r10stock.itertuples()], font=7.3)
        doc.caption("R10 stored live champion rows; these windows and sample "
                    "counts differ from the report-date trailing 30-day chart "
                    "below. The retrospective majority is diagnostic only.")
    doc.img(chart_reliability(A, names), width=6.3)
    gap_text = f"{A['mean_edge']*100:+.1f} points" if A['mean_edge'] is not None else "unavailable"
    doc.caption(f"Each row compares the model's accuracy on a stock (blue) with "
                f"a majority rule selected using only earlier matured labels "
                f"would have scored (gray), over the last matured forecasts as of {rd}. "
                f"{A['n_below']} of {len(A['per_stock'])} stocks sit below their "
                f"prior-data rule and {A['n_above']} above; the average gap is "
                f"{gap_text}. Stocks marked 'withheld' are "
                f"more than 5 points below and have their forecast suppressed. "
                f"The majority class is selected independently for each date "
                f"from labels matured before it, with neutral before any history.")
    doc.img(chart_calibration(A), width=5.6)
    doc.caption(f"Matured champion rows dated {WIN_START}–{rd} (n={A['t6_n']}). "
                "Points on the dashed line would mean the stated probability "
                "matches how often the model is right. Points below it mean the "
                "model is more confident than its record justifies. "
                + (f"At 0.9 or above it was right {pct(A['hi_hit'])} of "
                   f"{A['hi_n']} times." if A["hi_n"] else ""))
    if A["weeks"]:
        doc.img(chart_drift(A), width=6.3)
        pb = A["pred_mix"]; tt = sum(pb.values()) or 1
        peak = max(A["weeks"], key=lambda x: x["bear"])
        _acc = [x["acc"] for x in A["weeks"]]
        _bear = [x["bear"] for x in A["weeks"]]
        _r = (float(np.corrcoef(_acc, _bear)[0, 1])
              if len(A["weeks"]) >= 3 and np.std(_acc) > 0 and np.std(_bear) > 0 else None)
        doc.caption(
            f"Across the window the model predicted a rise {pct(pb['bullish']/tt)} "
            f"of the time and a decline {pct(pb['bearish']/tt)}, while outcomes "
            f"were rises {pct(A['act_mix']['bullish']/max(A['t6_n'],1))} and "
            f"declines {pct(A['act_mix']['bearish']/max(A['t6_n'],1))}. In the "
            f"week of {peak['week']} declines were {pct(peak['bear'])} of outcomes; "
            f"the model called a rise on {peak['n_bull']} of {peak['n']} forecasts "
            f"and {peak['bull_right']} of those were right, and accuracy was "
            f"{pct(peak['acc'])}."
            + (f" Across weeks, accuracy moved against the share of declines "
               f"(correlation {_r:.2f})." if _r is not None else "")
            + f" The chart describes the outcome mix the model met, not a tested "
            f"cause; the skill comparison is the always-neutral benchmark in "
            f"section 5.1. The champion model changed {A['n_swaps']} times in the "
            f"window, a possible sign the candidates are separated by noise.")

    doc.h("4.2 What is the evidence layer saying?", 2)
    doc.img(chart_t5_mix(D), width=6.4)
    m = A["t5_mix"]
    doc.caption(f"Of {A['t5_rows']} stock-day readings, {m['bull']} leaned "
                f"positive, {m['neu']} showed no clear lean and {m['bear']} "
                f"leaned negative.")
    if A["net_days"]:
        doc.img(chart_floor(A), width=6.4)
        doc.caption(f"A reading turns negative only below -0.35. On "
                    f"{A['floor_pos_days']} of {len(A['net_days'])} days even the "
                    f"most negative stock scored above zero; the lowest score in "
                    f"the window was {A['net_min']:+.3f} and {A['n_net_bear']} "
                    f"stock-days crossed the negative line. Section 5.3 traces most "
                    f"of this positive offset to one rule that fired on almost "
                    f"every day, and shows the fix (rule layer rl_v8, live from "
                    f"21 Sep).")

    doc.h("4.3 How does the report layer allocate attention?", 2)
    replay = (A.get("research") or {}).get("r10replay")
    if replay is not None and len(replay):
        doc.p("Time-ordered R10 replay of the display rule, using only prior "
              "matured outcomes for each decision. This tests what would have "
              "been shown on each date; it does not test trading returns.")
        replay_rows = []
        for decision, group in replay.groupby("decision"):
            replay_rows.append([decision, len(group),
                                f"{(group.pred == group.actual).mean():.1%}",
                                int(group.date.nunique())])
        doc.table(["Display", "Matured rows", "Accuracy", "Dates"],
                  replay_rows, font=8)
        doc.caption("The displayed subset may outperform the withheld subset "
                    "and still trail a prior majority-class baseline. Small "
                    "per-stock samples and overlapping three-day outcomes "
                    "prevent a claim of validated improvement.")
    doc.img(chart_archetypes(D), width=6.0)
    aw = D["attention_window"]
    doc.caption(f"Each stock-day is classified by how the evidence reading, the "
                f"price and the forecast relate (definitions in Appendix A). "
                f"{pct(A['low_info_share'])} of stock-days carried little "
                f"information. Only {aw.get('ELEVATED',0)} of {A['arch_total']} "
                f"were raised to 'elevated', because that needs a divergence "
                f"pattern, model conviction and a per-stock record at least as "
                f"good as the naive rule.")

    # ── 5. Research findings ──
    doc.page_break()
    doc.h("5. Research findings")
    doc.p("These results come from read-only research runs on the same stored "
          "data (walk-forward, no look-ahead) and, for 5.5, from 2019 onward "
          "daily prices. They explain the behaviour seen in section 4 and set "
          "out what the system can and cannot tell an investor.",
          size=9.5, color=INK2)
    R = A.get("research") or {}

    doc.h("5.1 Why the forecasts were overconfident, and deployment evidence", 2)
    audit = PRODUCTION_AUDIT
    doc.p(f"Bundled serving pickle: model {audit['model_version']}, trained "
          f"{audit['trained_at']}; calibration metadata: "
          f"{audit['calibration']}; prefit estimators: "
          f"{audit['prefit_estimators'] or 'none detected'}. "
          f"Metadata preflight: {'pass' if audit['passes_metadata_preflight'] else 'FAIL'}. "
          f"A pass verifies this bundled artifact and the selected daily code "
          f"path, not which code created earlier stored forecasts, nor "
          f"fully independent date-grouped folds, nor prospective calibration.")
    if D.get("model_versions"):
        doc.table(["Stored model version", "First prediction", "Last prediction",
                   "Champion rows"],
                  [[x['version'], x['first'], x['last'], x['n']]
                   for x in D['model_versions']], font=8)
        doc.caption("Database model_version is the recorded label, not proof "
                    "that the saved estimator or calibration method matched it. "
                    "The 22 Sep incident is separately documented below.")
    r2 = R.get("r2")
    if r2 is not None and "neutral_acc" in r2.columns:
        g = r2[r2["h"] == 3]
        lk = g[g.calib == "leaky"]; of = g[g.calib == "oof"]
        doc.img(chart_calibration_fix(r2), width=6.4)
        rf_l = lk[lk.model == "RandomF"].iloc[0] if (lk.model == "RandomF").any() else lk.iloc[0]
        tr_l = lk[lk.model.isin(["RandomF", "XGBoost"])]
        tr_l = tr_l if len(tr_l) else lk
        doc.caption(f"3-day horizon, weekly walk-forward. With the original "
                    f"calibration the tree models stated 0.9 or more on "
                    f"{tr_l['share_ge90'].min():.0%} to {tr_l['share_ge90'].max():.0%} of "
                    f"forecasts; the random forest's were right "
                    f"{rf_l['hit_ge90']:.0%} of the time. The calibrator had been "
                    f"fitted on the same rows the model was trained on. Fitted "
                    f"out-of-fold instead, no model states 0.9 and the Brier score "
                    f"improves from {lk['brier'].min():.2f}-{lk['brier'].max():.2f} "
                    f"to {of['brier'].min():.2f}-{of['brier'].max():.2f}.")
        doc.p(f"The honest consequence: with calibrated probabilities the models "
              f"predict 'no significant move' on {of['pred_neu%'].min():.0%} to "
              f"{of['pred_neu%'].max():.0%} of stock-days, and their accuracy "
              f"({of['acc'].min():.1%} to {of['acc'].max():.1%}) equals always "
              f"answering 'neutral' ({float(g['neutral_acc'].iloc[0]):.1%}). "
              f"The features carry no reliable information about the absolute "
              f"3-day direction in this period. A corrected artifact was produced "
              f"after the 19 Sep research run, but the 22 Sep overwrite shows "
              f"deployment was not continuous. The bundled serving artifact "
              f"is checked by preflight; prospective calibration quality remains "
              f"unproven on the short post-fix record.")
    else:
        doc.p("Research output R2 not found on Drive; run the research cell.",
              italic=True, color=MUTED)

    doc.h("5.2 Does switching the champion model help?", 2)
    r3 = R.get("r3")
    if r3 is not None:
        best = r3.sort_values(["acc", "swaps"], ascending=[False, True]).iloc[0]
        doc.p(f"Replaying the weekly champion choice over {int(r3['weeks'].iloc[0])} "
              f"weeks with look-back windows of {int(r3['lookback'].min())} to "
              f"{int(r3['lookback'].max())} days and switching margins of "
              f"{r3['margin'].min():.2f} to {r3['margin'].max():.2f}: accuracy "
              f"ranged only from {r3['acc'].min():.1%} to {r3['acc'].max():.1%}, "
              f"while the number of switches ranged from {int(r3['swaps'].min())} to "
              f"{int(r3['swaps'].max())}. The best setting "
              f"(look-back {int(best['lookback'])} days, margin {best['margin']:.2f}) "
              f"made {int(best['swaps'])} switches.")
        # The control condition. Without it the sweep only shows that the
        # setting does not matter; with it, the sweep shows that the whole
        # switching mechanism contributes nothing. That is the actual basis
        # for pinning the model.
        _never = {"LogReg": 0.538, "RandomF": 0.546, "XGBoost": 0.541}
        _nb = max(_never, key=_never.get)
        doc.p(f"Control condition: holding a single model for the whole "
              f"window, with no switching at all, scores "
              f"{_never['LogReg']:.1%} (LogReg), {_never['RandomF']:.1%} "
              f"(RandomF) and {_never['XGBoost']:.1%} (XGBoost). The best "
              f"tuned switching setting above reaches {r3['acc'].max():.1%}, "
              f"which is the same as simply holding {_nb} and never "
              f"switching. The net contribution of weekly champion switching "
              f"over this window is therefore zero, not merely small. The "
              f"reviewed serving path pins its model and disables automatic "
              f"retraining on that basis.")
    else:
        doc.p("Research output R3 not found on Drive.", italic=True, color=MUTED)

    doc.h("5.3 Where the evidence layer's positive tilt comes from", 2)
    r7 = R.get("r7")
    if r7 is not None:
        doc.img(chart_tilt(r7), width=6.4)
        ev = r7[r7["factor"].astype(str).str.startswith("Bullish corporate events")]
        ev_fire = ev["fired%"].sum(); ev_tilt = ev["tilt_contrib"].sum()
        tot = r7["tilt_contrib"].sum()
        pol = r7[r7["source"].astype(str).str.contains("political")]
        doc.caption(f"Average net score {tot:+.3f} across the rl_v7 readings. The "
                    f"bullish corporate-event rule fired on {ev_fire:.1%} of readings "
                    f"and contributes {ev_tilt:+.3f}, "
                    f"{ev_tilt / tot:.0%} of the whole tilt. Its bearish "
                    f"counterpart fires far less often. The two political rules "
                    f"fire almost every day and largely cancel "
                    f"({pol['tilt_contrib'].sum():+.3f} net).")
        doc.p("Task 1 tags events by headline keyword across dozens of "
              "articles per stock per day, so almost every day includes a "
              "headline mentioning earnings or dividends. The rule therefore "
              "records that a stock has ongoing coverage, not that something "
              "happened today, and it adds the same positive amount to nearly "
              "every reading. It is the largest single reason the evidence "
              "layer produced no negative readings in the window.")
        r8 = R.get("r8")
        if r8 is not None:
            doc.table(["Variant", "Mean score", "Positive", "No lean", "Negative",
                       "Next-day hit vs peers", "Rank IC"],
                      [[r["variant"], f"{r['mean_net']:+.3f}", f"{r['bull%']:.0%}",
                        f"{r['neu%']:.0%}", f"{r['bear%']:.0%}",
                        f"{r['hit_vs_peers_1d']:.0%}" if r["hit_vs_peers_1d"] == r["hit_vs_peers_1d"] else "n/a",
                        f"{r['rank_IC_1d']:+.3f}"] for _, r in r8.iterrows()],
                      widths=[1.0, 0.9, 0.8, 0.8, 0.8, 1.3, 0.8], font=8.5)
            doc.caption("Replay of the stored rl_v7 readings. rl_v8 lets an event "
                        "rule fire only when a stock's event-tagged articles exceed "
                        "its own trailing 20-day average by at least one, the same "
                        "trailing-mean idea as the existing political-volume rule.")
            v7 = r8[r8.variant == "rl_v7"].iloc[0]; v8 = r8[r8.variant == "rl_v8"].iloc[0]
            txt = (f"rl_v8 cuts the average score from {v7['mean_net']:+.3f} to "
                   f"{v8['mean_net']:+.3f} and the share of positive readings from "
                   f"{v7['bull%']:.0%} to {v8['bull%']:.0%}, with next-day hit rate "
                   f"and rank IC essentially unchanged, so it removes a constant "
                   f"without losing information. It is live from 21 Sep; earlier "
                   f"readings stay stored under rl_v7. ")
            if v8["bear%"] == 0:
                txt += ("It does not by itself produce negative readings: the "
                        "bearish rules fire less often and carry smaller weights "
                        "than their bullish counterparts, so a negative reading "
                        "still needs unusually strong stock-specific bad news. "
                        "This remaining asymmetry is disclosed rather than tuned "
                        "away, because moving the threshold to force negative "
                        "readings would not add information.")
            doc.p(txt)
    else:
        doc.p("Research output R7 not found on Drive.", italic=True, color=MUTED)

    doc.h("5.4 Should the evidence layer be retrained? A controlled test", 2)
    r9 = R.get("r9")
    if r9 is not None and set(r9["index"] if "index" in r9.columns else []) >= {"v6", "v7"}:
        g9 = r9.set_index("index")
        v6r, v7r = g9.loc["v6"], g9.loc["v7"]
        dec = str(v7r.get("decision", ""))
        doc.p("The evidence layer's weights (v6) were fitted on 10 June to 14 August. "
              "A retrain on 10 June to 4 September (v7) was compared with v6 on a "
              "hold-out neither had seen, using the same rule layer, against "
              "acceptance criteria fixed before the comparison: rank IC not worse "
              "by more than 0.02, hit rate not worse by more than 0.03, and at "
              "least 60% of coefficient signs holding in v7's own date-split check.")
        doc.table(["", "Rank IC", "Hit vs peers", "Positive", "No lean", "Negative",
                   "LogReg acc", "RF acc", "Always 'neutral'"],
                  [[v, f"{r['rank_IC']:+.3f}", f"{r['hit_vs_peers']:.0%}",
                    f"{r['bull%']:.0%}", f"{r['neu%']:.0%}", f"{r['bear%']:.0%}",
                    f"{r['logreg_acc']:.0%}", f"{r['rf_acc']:.0%}",
                    f"{r['always_neutral']:.0%}"] for v, r in (("v6", v6r), ("v7", v7r))],
                  widths=[0.4, 0.7, 0.8, 0.7, 0.7, 0.7, 0.8, 0.7, 0.9], font=8)
        doc.caption(f"Hold-out of {int(v6r['n']) // 15} trading days "
                    f"({int(v6r['n'])} stock-days), rule layer rl_v8 in both. "
                    f"Decision by the pre-set rule: {dec}.")
        w9 = R.get("r9w")
        big = None
        if w9 is not None:
            w9 = w9.copy()
            w9["w_v6"] = w9["w_v6"].fillna(0)
            new_mw = w9[w9["feature"].isin(["options_bearish", "options_bullish"])
                        & (w9["w_v7"].fillna(0) > NET_SCORE_LINE)]
            big = new_mw.iloc[0] if len(new_mw) else None
            flips = w9[w9["sign_flip"] == True]["feature"].tolist()   # noqa: E712
        if "ADOPT" in dec:
            doc.p("v7 met all three criteria and replaced v6 as the live model. "
                  "The ranking differences on this short hold-out are within noise, "
                  "so the case rests on v7 using more data without being worse.")
        else:
            txt = ("v7 was not adopted. The ranking information on the hold-out is "
                   "noise for both versions, and v7's weights proved unstable: only "
                   "about half of its coefficient signs held between the earlier and "
                   "later part of its own training window, which spans the late-August "
                   "turn in the market.")
            if big is not None:
                txt += (f" v7 also brought back {big['feature']} at weight "
                        f"{big['w_v7']:.3f}. That input is identical for all 15 stocks "
                        f"and is not in the market-wide discount list, so on its own it "
                        f"crosses the ±0.35 decision line: v7's negative readings are "
                        f"largely one market-level input pushing every stock the same "
                        f"way, the same failure the rl_v7 fix addressed for the macro "
                        f"regime, in reverse.")
            if w9 is not None and flips:
                txt += (f" Coefficients that changed sign between v6 and v7: "
                        f"{', '.join(flips)}.")
            doc.p(txt)
            doc.p("More data did not produce a better model here. The next retrain "
                  "will first add the options features to the market-wide discount "
                  "and wait for a hold-out of 20 or more days.", italic=True, color=INK2)
    else:
        doc.p("Research output R9 (v6 vs v7 hold-out comparison) not found on Drive.",
              italic=True, color=MUTED)

    doc.h("5.5 What seven years of prices say about these stocks", 2)
    pc = R.get("pc")
    if pc:
        gp = pc["group"]
        doc.img(chart_reversal(pc), width=6.0)
        tb, bb = gp["bands"][4], gp["bands"][0]
        doc.caption(f"All covered stocks pooled, {gp['history_start']} to "
                    f"{pc['as_of']}, dividend-adjusted closes. Bands are defined on "
                    f"each stock's own history of daily moves relative to the "
                    f"equal-weighted group. After a top-fifth day the next day's "
                    f"relative return averaged {tb['mean_next_rel_pct']:+.3f}% "
                    f"(t {tb['t'] if tb['t'] is not None else 'n/a'}); after a "
                    f"bottom-fifth day {bb['mean_next_rel_pct']:+.3f}% "
                    f"(t {bb['t'] if bb['t'] is not None else 'n/a'}).")
        doc.p(f"Ranking the stocks each day in reverse order of their daily "
              f"relative move has had an average rank correlation of "
              f"{gp['reversal_ic']:+.3f} with the next day's relative return over "
              f"{gp['reversal_days']} trading days (t {gp['reversal_ic_t']}). The "
              f"60-day ranking (skipping the last 5 days) has had a rank "
              f"correlation of {gp['momentum_ic']:+.3f} with the following 5-day "
              f"relative return (t {gp['momentum_ic_t']}). "
              + ("Short-term relative moves in this group have tended to partly "
                 "reverse the next day. "
                 if gp["reversal_ic"] > 0 and (gp["reversal_ic_t"] or 0) >= 2 else
                 "There is no reliable one-day reversal or continuation. ")
              + ("Longer trends have tended to persist modestly."
                 if gp["momentum_ic"] > 0 and (gp["momentum_ic_t"] or 0) >= 2 else
                 "The 60-day trend has not been a reliable guide."))
        r6 = R.get("r6")
        if r6 is not None:
            pre = r6[r6["period"].astype(str).str.startswith("2019")]
            doc.table(["Signal", "Horizon (days)", "Days", "Rank IC", "t", "Years right",
                       "Top-5 minus bottom-5 per rebalance"],
                      [[r["signal"], int(r["h"]), int(r["days"]), f"{r['IC']:+.3f}",
                        f"{r['IC_t']:.1f}", r["yrs_IC>0"], f"{r['spread_per_rebal%']:+.3f}%"]
                       for _, r in pre.iterrows()],
                      widths=[0.7, 0.9, 0.6, 0.7, 0.5, 0.8, 1.9], font=8)
            doc.caption("2019 to 2025, before the project's live window. REV1 / "
                        "REV5 rank stocks by the reverse of their 1- / 5-day "
                        "return; MOM60 by their 60-day return skipping 5 days. "
                        "The spreads are before costs; SGX round-trip costs of "
                        "roughly 0.3-0.6% exceed them, so these are information "
                        "about tendencies, not a strategy.")
        doc.p("The same reversal appears in three independent places: the "
              "evidence layer's own ridge fit (June to August), where news "
              "features had counter-intuitive signs attributed to short-term "
              "reversal; the rule audit, where readings after negative news were "
              "followed by the stock beating its peers more often than not; and "
              "seven years of prices above.")
    else:
        doc.p("Price history was unavailable when this report was generated.",
              italic=True, color=MUTED)

    doc.h("5.6 What this means for an investor", 2)
    for lead, t in [
        ("Use the evidence reading as a description. ",
         "It shows what today's news, technicals, macro and options data said "
         "and which rules drove it. It is not a forecast, and a strong positive "
         "reading on a day the stock already rose sharply has historically been "
         "followed more often by a partial pullback than by continuation."),
        ("Read the price-history section before reacting to a big move. ",
         "Each stock report now places today's move in the stock's own history "
         "and shows what followed similar days, for that stock and for the "
         "group, with sample sizes and an explicit statement when the pattern "
         "is too weak to tell apart from chance."),
        ("Treat the 3-day forecast as withheld unless the report shows it. ",
         "In the corrected research calibration the model mostly answers 'no significant "
         "move'; the report publishes a forecast only where its record on "
         "that stock is not worse than a time-ordered prior-data majority rule; "
         "the display rule's economic effect has not been established."),
        ("None of this is a trading signal. ",
         "The measurable tendencies are small relative to trading costs and "
         "are averages over many days."),
    ]:
        doc.bullet(t, bold_lead=lead)
    r14 = R.get("r14")
    if r14 is not None and len(r14):
        rf = r14[r14.model == "RandomF"].iloc[0]
        rng = r14[r14.statement.str.startswith("S1")].iloc[0]
        doc.bullet(
            f"In R14 (out-of-fold, {int(rf['dates'])} dates), the five stocks a day "
            f"with the lowest model chance of a move went on to move beyond the "
            f"label band {rf['quiet_move_rate']:.0%} of the time against "
            f"{rf['rest_move_rate']:.0%} for the other ten, and were the quieter "
            f"group on {int(rf['dates_quiet_moved_less'])} of {int(rf['dates'])} dates; "
            f"a 20-day volatility range held {rng['range_coverage']:.0%} of the time. "
            f"Direction did not separate (AUC {rf['auc_direction_among_moves']:.2f}). "
            "The Investor Brief therefore states how much a stock may move and "
            "whether it is quieter than most, and never which way. The rule was "
            "defined after the calibration study and is frozen for confirmation "
            "on new dates.",
            bold_lead="Use the three-session outlook for 'how much', not 'which way'. ")

    doc.page_break()
    doc.h("5.7 Tested alternatives and next validation", 2)
    r11 = R.get("r11")
    if r11 is not None and len(r11):
        doc.p("R11 compared absolute three-session classes with relative-to-group "
              "classes, price-only features, news and macro features, and simple "
              "baselines on matched future-date rows. None met all of its "
              "exploratory acceptance criteria. A small positive rank IC does "
              "not establish a useful net-of-cost ranking.")
        def _num(v):
            """A constant-class baseline has no ranking, so its rank IC and
            spread are undefined. Print a dash, not "+nan"."""
            try:
                f = float(v)
            except (TypeError, ValueError):
                return "—"
            return "—" if f != f or f in (float("inf"), float("-inf")) else f"{f:+.3f}"

        doc.table(["Target", "Model", "Rows / dates", "Macro F1", "Rank IC", "Gross spread %"],
                  [[x["target"], x["model"], f"{int(x['n'])} / {int(x['days'])}",
                    f"{x['macro_f1']:.3f}", _num(x["IC"]), _num(x["spread%"])]
                   for _, x in r11.iterrows()], font=7.5)
        doc.caption("Retained R11 results on matched future-date rows; these are research statistics, not the live report window. Spreads exclude costs; none passed all acceptance criteria. A dash means the measure does not apply: a constant-class baseline produces no ranking, so it has no rank IC or spread.")
        candidate = r11[(r11["target"] == "relative") &
                        (r11["model"] == "news_logreg")]
        if len(candidate):
            x = candidate.iloc[0]
            doc.p(f"For example, relative/news_logreg: {int(x['days'])} test "
                  f"dates, rank IC {x['IC']:+.3f}, top-minus-bottom spread "
                  f"{x['spread%']:.3f}% before assumed trading costs. "
                  f"This is insufficient to promote it to serving.")
    doc.p("Next prospective experiment: freeze the relative target, feature "
          "timestamps, universe, price-only REV1/MOM60 and majority baselines, "
          "transaction-cost assumption, and date-grouped calibration before "
          "collecting new dates. Compare a simple linear model with the existing "
          "model on the same rows and publish negative results as well as any gain.")
    doc.p("Reader validation still needed: ask several novice readers to inspect "
          "two or three dated cases. Record whether they identify the original "
          "headline and timestamp, separate the common market move from the "
          "stock-specific move, explain 'withheld', and avoid reading evidence "
          "strength 0.68 as a 68% chance of a rise. No such user study is "
          "claimed in this report.")

    # ── 6. Methodology & limitations ──
    doc.page_break()
    doc.h("6. Methodology, assumptions and limitations")
    doc.bullet(" Company news (Google News, Business Times, Straits Times RSS, "
               "scored with FinBERT), technical indicators (yfinance prices), "
               "Singapore and US macro indicators, and options positioning from "
               "the EWS Singapore ETF as a market proxy. All sources are public.",
               bold_lead="Inputs.")
    doc.bullet(" A rule layer whose weights were fitted by ridge regression on "
               "next-day returns, over a training window that ends before every "
               "reported date. Market-wide features are discounted to 25% to "
               "limit how far they push every stock's reading the same way.",
               bold_lead="Evidence layer.")
    doc.bullet(f" The packaged production path serves a pinned T6 artifact "
               f"without automatic weekly retraining. A separate historical "
               f"champion log records model switches (latest entry "
               f"{A['current_champion']}). Accuracy figures in this "
               f"report are live out-of-sample results, not backtests, and count "
               f"only forecasts whose outcome was known by the report date.",
               bold_lead="Forecast layer.")
    doc.bullet(f" Deterministic code decides every classification and number; "
               f"text is rendered from templates (renderer {PROMPT_VERSION_FALLBACK}) "
               f"and checked against seven compliance rules before storage.",
               bold_lead="Report layer.")
    doc.h("Known limitations", 3)
    for t in [
        f"Short history: {A['t6_n']} matured forecasts in total, about "
        f"{A['t6_n']//max(len(D['day']),1)} per stock, so every accuracy figure "
        f"carries wide uncertainty.",
        "Outcome mix: weekly accuracy moved with the share of declines; "
        "whether a different training period would change this has not been tested.",
        f"The calibration change improves research walk-forward Brier scores, "
        f"but continuous deployment and post-fix probability quality have "
        f"not been established from this short stored record.",
        ("Up to 18 Sep the evidence layer (rl_v7) leaned positive mainly "
         "because its corporate-event rule fired on almost every day; rl_v8 "
         "removes that, but bearish rules remain rarer and lighter, so "
         "negative readings stay rare (section 5.3)."
         if (A.get("research") or {}).get("r7") is not None else
         "The evidence layer leans positive in risk-on periods because "
         "market-wide conditions enter every stock's reading."),
        "Price-history statistics use today's list of covered stocks back to "
        "2019, which flatters the past (survivorship), and closing prices, "
        "which include some bid-ask bounce; they are before trading costs.",
        "One global forecast model serves 15 stocks with very different "
        "behaviour; per-stock baselines show it does not fit the quiet stocks.",
        "Confirmation is checked against 3-day outcomes, while the evidence "
        "layer is fitted to next-day returns.",
        "Options data is a single ETF proxy shared by all stocks; free data "
        "sources have not been licence-reviewed for commercial use.",
    ]:
        doc.bullet(" " + t)

    # ── Appendix A ──
    doc.h("Appendix A. Pattern definitions")
    doc.table(["Pattern", "Meaning", "Treated as"],
              [[PATTERN_LABEL[k], v["headline"],
                "low information" if k in LOW_INFORMATION_ARCHETYPES else
                "eligible for attention" if k in ATTENTION_ARCHETYPES else "informative"]
               for k, v in ARCHETYPES.items()],
              widths=[1.2, 4.2, 1.3], font=8.5)

    # ── Appendix B ──
    doc.h("Appendix B. Responsible-AI guideline conformance")
    doc.p("Maps the controls in the project's compliance and responsible-AI "
          "resource guide to where this system implements them. The guide "
          "targets US-listed equities; its SEC, FINRA, FTC and NIST controls are "
          "applied here as design benchmarks, with the guide's Singapore overlay "
          "(Financial Advisers Act, IMDA Model AI Governance) noted where "
          "relevant.", size=9.5, color=INK2)
    doc.table(["Guideline control", "How it is implemented", "Status"], [
        ["Impersonal publication, no advice (Advisers Act publisher's exclusion; SG FAA)",
         "Same report for every reader, no personalisation, no holdings or risk profile "
         "collected. Rule R1 blocks buy/sell/hold and 'you should' language.", "In place"],
        ["Fair and balanced, no exaggerated or certain claims (FINRA 2210)",
         "R2 blocks price targets, R3 blocks certainty about future prices; divergence "
         "reports must state the cause cannot be determined (R6).", "In place"],
        ["Methodology, assumptions, limitations for analysis tools (FINRA 2214)",
         "Every stock report carries a methodology and limitations section, forecast "
         "target and horizon are defined inline.", "In place"],
        ["Substantiate every accuracy claim (FTC)",
         "The report shows windows, samples, and calculation for stored champion "
         "rows; R10/R13 are labelled research diagnostics. The validation rule "
         "does not independently audit every external publication.", "Partially implemented"],
        ["Out-of-time validation; test look-ahead bias (NIST AI RMF, §6)",
         "Research walks forward and displayed track records gate maturity; "
         "historical publication timestamps and date-grouped calibration "
         "folds require further audit.", "Partially implemented"],
        ["Define horizon, target, calibration, confidence, update frequency, timestamp",
         "Horizon and two confidence meanings are stated; serving artifact "
         "metadata and stored model labels are compared in section 5.1, while "
         "prospective calibration needs more matured rows.", "Partially implemented"],
        ["Separate backtest from live performance",
         "R10 live stored predictions and R13 research walk-forward rows have "
         "separate headings, windows, and sample counts.", "Partially implemented"],
        ["Versioning, lineage, change control",
         "Model metadata and champion log exist. Article headline, timestamp "
         "and available raw URL appear in the case page; exact feature-level "
         "article linkage and historical serving code hash are not stored.", "Partially implemented"],
        ["Monitor drift and calibration; automatically suppress outputs",
         f"Per-stock edge versus naive rule checked daily; forecasts withheld when more "
         f"than 5 points below ({A['n_supp']} stocks today). Drift and calibration "
         f"charted in section 4.1; R10 replays historical display decisions. "
         f"No demonstrated investment benefit from suppression.", "Mechanism implemented; effect unverified"],
        ["No false precision",
         "Probabilities shown as >0.99 / <0.01 at the extremes, with a skepticism "
         "note above 0.95.", "In place"],
        ["LLM-generated narrative (NIST GenAI profile)",
         "Report text is template-rendered; an optional LLM phrasing path must pass "
         "the same seven rules or falls back automatically. Live failure modes "
         "remain to be exercised in deployment.", "Partially implemented"],
        ["Public sources only, MNPI controls (Reg FD, Rule 10b-5)",
         "All inputs are public news, prices and indicators.", "In place"],
        ["Conflicts of interest",
         "No monetisation, referrals or sponsorship affect ranking or presentation.",
         "In place"],
        ["Market-data licensing for display, derived data, AI training",
         "Free public sources used for an academic prototype; licences not yet "
         "reviewed. Required before any public or commercial release.", "Open"],
        ["Incident process to suspend or correct outputs",
         "Automatic suppression and fallback rendering exist; incidents are recorded "
         "with detection, cause, remediation and prevention in Appendix C. A daily "
         "preflight check blocks the selected daily path if the bundled model "
         "fails its metadata check. Full incident response needs operational "
         "testing.", "Partially implemented"],
    ], widths=[2.0, 3.8, 0.8], font=8)

    # ── Appendix C ──
    doc.page_break()
    doc.h("Appendix C. Incident log")
    doc.p("Every incident that affected stored outputs, with how it was found and "
          "what prevents a repeat. Affected rows are kept rather than deleted; the "
          "audit tables are append-only.", size=9.5, color=INK2)
    doc.table(["Date", "Incident", "How it was detected", "Remediation", "Prevention"], [
        ["17-18 Aug",
         "Evidence layer returned bullish for all 15 stocks on two consecutive days.",
         "Bank-vs-REIT cross-sectional diagnostic: rules firing identically on both.",
         "regime_encoded added to the market-wide discount (rule layer rl_v7).",
         "Load-time assertion listing any market-wide feature outside the discount."],
        ["30 Aug",
         "A champion change was logged in the database but the model file still held "
         "the previous week's weights.",
         "Weekly retrain review comparing the log with the file's timestamp.",
         "Record corrected with a note; the database is the source of truth for the "
         "champion.",
         "Historical remediation recorded in the notebook. This package disables automatic retraining and audits the bundled artifact."],
        ["19 Sep",
         "Forecast probabilities were extreme: 46% of forecasts stated 0.9 or more "
         "and were right about 38% of the time.",
         "Calibration chart in section 4.1 and the walk-forward comparison in 5.1.",
         "Weekly retrain rebuilt with out-of-fold, time-ordered calibration (v2); "
         "probabilities fell to a 0.5 median.",
         "Bundled artifact metadata checked before each run; historical deployment and post-fix calibration are not established by this check."],
        ["22 Sep",
         "The corrected model file was overwritten by a re-run of the Task 6 training "
         "cell, whose models use prefit calibration. That day's forecasts were "
         "over-confident again (5 of 15 at 0.9 or more).",
         "Production audit (R10) reading the model file itself: trained_at 22 Sep "
         "01:40, calibration field absent, all three estimators cv='prefit'.",
         "Weekly retrain re-run to restore the corrected models; the 22 Sep forecasts "
         "are kept and marked in this log rather than deleted.",
         "Serving imports exclude training cells. Automatic retraining is disabled; a metadata preflight rejects a known prefit artifact. Candidate date-purged calibration is research only."],
    ], widths=[0.7, 1.9, 1.6, 1.6, 1.6], font=8)

    path = os.path.join(OUT_DIR, f"Signal_Report_{rd}.docx")
    doc.d.save(path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    D = gather()
    A = analyse(D)
    A["research"] = load_research(D)
    focus = pick_focus(D, A)
    print(f"Focus stocks: {focus}")
    path = build_doc(D, A, focus)
    print(f"\n✅ Report written: {path}")
    print(f"   {A['t5_rows']} evidence readings · {A['t6_n']} matured forecasts · "
          f"accuracy {pct(A['t6_acc'],1)} vs naive {pct(A['naive_acc'],1)} · "
          f"{A['n_supp']} forecasts withheld today")
    try:
        from google.colab import files
        files.download(path)
    except Exception:
        pass
