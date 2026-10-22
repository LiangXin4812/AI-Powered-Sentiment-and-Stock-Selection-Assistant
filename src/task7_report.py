"""
task7_report.py — Responsible AI report layer

Fuses Task 5 (evidence assessment) and Task 6 (3-day forecast) into a daily
digest. This module holds the deterministic parts: archetype classification,
reliability metrics, attention allocation, and source attribution. The LLM
only phrases what these functions decide.

Written as a module rather than a notebook cell so it can be imported from
Colab now and containerised later without changes:

    import sys; sys.path.append('/content/drive/MyDrive/sgx_project/src')
    from task7_report import build_report_input, classify_archetype

DESIGN DECISIONS THIS FILE ENCODES
----------------------------------
1. Reports are generated AFTER market close. All three inputs are then
   available: T5's reading of the day, T6's forward forecast, and the day's
   realised return. Users read the next morning.

2. Archetype is decided by code, never by the LLM — same principle as Task 5,
   where the LLM writes but does not decide.

3. Ten archetypes on two axes, not an enumeration of cases. Axis one is
   whether the day's price confirmed T5. Axis two is whether T6 continues or
   reverses T5. A tenth archetype covers T5 being neutral to begin with.

4. "No reaction" is separated from "went the other way". Roughly half of all
   days land inside the neutral band, and collapsing those into "not
   confirmed" would flood the interesting archetypes with quiet days.

5. Attention is scarce. Most stocks on most days should be ROUTINE. Three
   conditions must hold together before a stock is ELEVATED.

6. Reliability is reported per stock and gates the language. A stock where
   the model has underperformed the majority baseline must not produce
   confident phrasing regardless of today's numbers.

7. Source attribution is a first-class output. T5 and T6 both consume all of
   Tasks 1-4 — news, technical, macro, options. A report that mentions only
   news would misrepresent the system, so every report carries a breakdown of
   which sources actually drove the score.

8. T5 and T6 use different label thresholds because they label different
   horizons (next-day vs 3-day). Using the wrong one silently distorts the
   archetype distribution.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

_SGT = timezone(timedelta(hours=8))

import numpy as np
from sqlalchemy import text


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# T5 labels next-day returns; Part A v6 recomputed this at ±0.604% on 585
# samples. T6 labels 3-day returns at ±1.09%. Applying T6's threshold to T5
# would push many confirmed days into the "no reaction" row.
T5_THRESHOLD_DEFAULT = 0.604
T6_THRESHOLD_DEFAULT = 1.09

# Three-class majority baseline from Task 6's training distribution. A model
# scoring below this is not beating "always predict the majority class".
MAJORITY_BASELINE = 0.517

TRACK_RECORD_WINDOW = 30    # calendar days

# ELEVATED requires all three. Tuned so 1-3 of 15 stocks flag on a typical day.
ATTENTION_MIN_T6_PROB   = 0.55   # model has some conviction
ATTENTION_MIN_T6_ACC    = MAJORITY_BASELINE   # model has earned trust here
# lag / contradicted  — evidence and price disagree
# model_only          — evidence silent, model speaks
# All still have to clear the probability and track-record gates below.
ATTENTION_ARCHETYPES    = {"lag", "contradicted", "model_only"}

SOURCE_META = {
    "task1_news":      ("News",      "per-stock"),
    "task1_political": ("Political", "market-wide"),
    "task2_technical": ("Technical", "per-stock"),
    "task3_macro":     ("Macro",     "market-wide"),
    "task4_options":   ("Options",   "market-wide"),
}


# ═══════════════════════════════════════════════════════════════════════════
# ARCHETYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

ARCHETYPES = {
    "price_unavailable": {
        "axis": ("unavailable", "unavailable"),
        "headline": "Same-date price context is unavailable",
        "note": "The evidence cannot be compared with a same-date closing move.",
    },
    "momentum": {
        "axis": ("confirmed", "continue"),
        "headline": "Signals and price agreed; model expects the move to continue",
        "note": "Evidence, today's price and the forecast all point the same way.",
    },
    "mean_reversion": {
        "axis": ("confirmed", "reverse"),
        "headline": "Signals and price agreed, but the model expects a pullback",
        "note": ("The move happened, and the model reads the setup as stretched. "
                 "Historically this pattern appears after a signal has been "
                 "absorbed into the price."),
    },
    "move_complete": {
        "axis": ("confirmed", "flat"),
        "headline": "Signals played out today; nothing further expected",
        "note": "The reading was confirmed and the model sees no follow-through.",
    },
    "muted_pending": {
        "axis": ("muted", "continue"),
        "headline": "Signals leaned one way but price barely moved; model still expects that direction",
        "note": ("The evidence has not shown up in the price yet. The model "
                 "expects it to over the next three days."),
    },
    "muted_fading": {
        "axis": ("muted", "reverse"),
        "headline": "Signals leaned one way, price barely moved, model expects the opposite",
        "note": ("The evidence did not translate into price action and the "
                 "model now leans against it."),
    },
    "dormant": {
        "axis": ("muted", "flat"),
        "headline": "Signals had a slight lean; neither price nor model followed",
        "note": "Weak evidence, no price reaction, no forecast conviction.",
    },
    "lag": {
        "axis": ("contradicted", "continue"),
        "headline": "Price went against the signals, but the model still sides with the signals",
        "note": ("Today's price moved opposite to what the signals suggested, "
                 "yet the model expects the signalled direction over three days."),
    },
    "contradicted": {
        "axis": ("contradicted", "reverse"),
        "headline": "Price went against the signals and the model agrees with the price",
        "note": ("Both today's price and the forecast disagree with the signal "
                 "reading. The signals may be missing what the market is "
                 "reacting to."),
    },
    "noise": {
        "axis": ("contradicted", "flat"),
        "headline": "Price went against the signals; model has no clear view",
        "note": "The parts of the system disagree and none of them is confident.",
    },
    "quiet": {
        "axis": ("neutral", "flat"),
        "headline": "No directional signal today",
        "note": "Neither the evidence nor the model found a lean.",
    },
    "model_only": {
        "axis": ("neutral", "directional"),
        "headline": "The signals were flat, but the model has a view",
        "note": ("Today's news and indicators did not lean either way, yet "
                 "the model reads a direction from the combination of "
                 "technical, macro and options data. This is the kind of "
                 "pattern that does not appear in headlines."),
    },
}

# Archetypes that never warrant attention regardless of other conditions
# model_only is deliberately absent: it is the case where the model adds
# something the reader could not have seen themselves.
LOW_INFORMATION_ARCHETYPES = {"quiet", "dormant", "move_complete", "noise"}


def classify_actual(return_pct, threshold=None):
    """Realised return → class, using the currently configured threshold."""
    if threshold is None:
        threshold = T5_THRESHOLD_DEFAULT
    if return_pct is None:
        return None
    if return_pct > threshold:
        return "bullish"
    if return_pct < -threshold:
        return "bearish"
    return "neutral"


def classify_archetype(t5_sentiment, actual_class, t6_class, t6_prob=None):
    """
    Deterministic ten-way classification. The LLM never decides this.

    Axis 1 — did today's price confirm T5?
        confirmed     price moved the way T5 leaned
        muted         price stayed inside the neutral band
        contradicted  price moved against T5

    Axis 2 — what does T6 do with T5's direction?
        continue      same direction
        reverse       opposite direction
        flat          T6 says neutral

    Returns (archetype_key, axis1, axis2).
    """
    if t5_sentiment is None or t6_class is None:
        return "incomplete", None, None

    if t5_sentiment == "neutral":
        # A neutral evidence reading is not automatically uninteresting. If
        # the model still has a directional view, that is the model seeing
        # structure the rule layer missed — worth surfacing separately.
        if t6_class == "neutral":
            return "quiet", "neutral", "flat"
        # model_only needs conviction: a 0.4-probability "view" is a coin flip
        # with a direction attached. Gate matches ATTENTION_MIN_T6_PROB.
        # Measured 17-21 Aug: 13 of 14 model_only cases passed it anyway.
        if t6_prob is not None and t6_prob < ATTENTION_MIN_T6_PROB:
            return "quiet", "neutral", "flat"
        return "model_only", "neutral", "directional"

    if actual_class is None:
        return "price_unavailable", None, None
    elif actual_class == t5_sentiment:
        axis1 = "confirmed"
    elif actual_class == "neutral":
        axis1 = "muted"
    else:
        axis1 = "contradicted"

    if t6_class == "neutral":
        axis2 = "flat"
    elif t6_class == t5_sentiment:
        axis2 = "continue"
    else:
        axis2 = "reverse"

    for key, meta in ARCHETYPES.items():
        if meta["axis"] == (axis1, axis2):
            return key, axis1, axis2

    return "incomplete", axis1, axis2


# ═══════════════════════════════════════════════════════════════════════════
# TRACK RECORD
#
# These are retrospective by construction. That does not make them useless:
# they are not signals, they are the reliability of the signals. A weather
# forecaster's historical hit rate cannot predict tomorrow, but it tells you
# how much to trust tomorrow's forecast.
# ═══════════════════════════════════════════════════════════════════════════

def compute_track_record(engine, ticker, as_of_date,
                         window_days=None,
                         t5_threshold=None):
    """
    Per-stock reliability over the trailing window.

    Returns:
        t5_coherence  T5's direction vs the SAME day's move.
                      Does the evidence read the present correctly?
        t5_lead       T5's direction vs the NEXT day's move.
                      Does the evidence lead price at all?
        t6_accuracy   T6 champion's 3-day directional accuracy.
        t6_f1         Weighted F1, robust to the class imbalance.

    coherence vs lead is worth reporting as a pair. T5's Ridge weights are
    fitted on next-day returns, so lead should exceed coherence if the layer
    has any predictive content. If coherence is higher, the layer is
    describing what already happened rather than anticipating anything —
    a finding worth stating plainly in the final report.
    """
    window_days = TRACK_RECORD_WINDOW if window_days is None else window_days
    t5_threshold = T5_THRESHOLD_DEFAULT if t5_threshold is None else t5_threshold
    out = {"t5_coherence": None, "t5_lead": None,
           "t6_accuracy": None, "t6_f1": None,
           "t5_n": 0, "t6_n": 0}

    # ── T5: join each reading to same-day and next-day returns ──
    with engine.connect() as conn:
        rows = conn.execute(text("""
            WITH t5 AS (
                SELECT ticker, analysis_date::date AS d, sentiment
                FROM llm_analyses
                WHERE ticker = :tkr
                  AND analysis_date::date <  CAST(:asof AS DATE)
                  AND analysis_date::date >= CAST(:asof AS DATE) - CAST(:win AS INTEGER)
            ),
            px AS (
                SELECT ticker, prediction_date::date AS d,
                       MAX(actual_return_3d) AS r3
                FROM task6_predictions
                WHERE ticker = :tkr AND actual_return_3d IS NOT NULL
                  AND forecast_target <= CAST(:asof AS DATE)
                GROUP BY ticker, prediction_date
            )
            SELECT t5.d, t5.sentiment, px.r3
            FROM t5 LEFT JOIN px ON t5.ticker = px.ticker AND t5.d = px.d
            ORDER BY t5.d
        """), {"tkr": ticker, "asof": str(as_of_date), "win": window_days}).fetchall()

    # NOTE: task6_predictions stores 3-day returns. Same-day returns for T5
    # coherence should come from a dedicated daily-return source. Until that
    # exists, coherence is computed against the 3-day return as a rough proxy
    # and flagged as such. Replace once a daily price table is available.
    graded = [(str(d), s, r) for d, s, r in rows if r is not None]
    if graded:
        # r is a three-day return (see note above) so it takes the
        # three-day threshold, not t5_threshold.
        hits = sum(1 for _, s, r in graded
                   if classify_actual(r, T6_THRESHOLD_DEFAULT) == s)
        out["t5_coherence"] = hits / len(graded)
        out["t5_n"] = len(graded)

    # ── T6: champion rows only ──
    # NOTE: ":asof::date" does not work. SQLAlchemy's named-parameter parser
    # reads ":asof:" as the parameter name and leaves the rest as literal SQL,
    # producing a syntax error at the colon. Use CAST(:param AS DATE) instead.
    # Same reason the day offset is an INTERVAL rather than "- :win".
    with engine.connect() as conn:
        t6 = conn.execute(text("""
            SELECT DISTINCT ON (prediction_date) prediction_date::text,
                   forecast_target::text, predicted_class, actual_class
            FROM task6_predictions
            WHERE ticker = :tkr
              AND is_champion = TRUE
              AND actual_class IS NOT NULL
              -- Look-ahead guard: a prediction counts toward the track record
              -- only if its 3-day outcome was already known on the as-of
              -- date. Without this, a report dated in the past quotes
              -- accuracy using prices from after that date.
              AND forecast_target <= CAST(:asof AS DATE)
              AND prediction_date <  CAST(:asof AS DATE)
            ORDER BY prediction_date, id DESC
        """), {"tkr": ticker, "asof": str(as_of_date), "win": window_days}).fetchall()

    from sgx_capstone.benchmarks import prior_majority_labels
    from datetime import timedelta
    start = (datetime.strptime(str(as_of_date), "%Y-%m-%d") -
             timedelta(days=window_days)).date().isoformat()
    history = [dict(date=str(r[0]), maturity=str(r[1]), pred=r[2], actual=r[3]) for r in t6]
    evaluation = [r for r in history if r["date"] >= start]
    if evaluation:
        pred = [r["pred"] for r in evaluation]
        act  = [r["actual"] for r in evaluation]
        baseline = prior_majority_labels(history, evaluation)
        out["t6_n"] = len(evaluation)
        out["t6_accuracy"] = sum(p == a for p, a in zip(pred, act)) / len(evaluation)
        out["window_days"] = window_days

        # Per-stock majority-class baseline: what "always guess this stock's
        # most common outcome" would have scored over the same window. This
        # is the bar the model has to clear to be worth reading.
        from collections import Counter as _Counter
        _cnt = _Counter(act)
        out["t6_baseline"] = sum(p == a for p, a in zip(baseline, act)) / len(act)
        out["t6_baseline_class"] = "prior-data majority; neutral before any matured history"
        out["t6_diagnostic_majority"] = _cnt.most_common(1)[0][1] / len(act)
        out["t6_baseline_kind"] = "time-ordered prior matured labels"
        out["t6_edge"]           = out["t6_accuracy"] - out["t6_baseline"]
        try:
            from sklearn.metrics import f1_score
            out["t6_f1"] = f1_score(act, pred, average="weighted", zero_division=0)
        except ImportError:
            pass

    return out


def reliability_verdict(track):
    """
    Turn reliability numbers into a language gate.

    'insufficient'  too few matured samples to say anything
    'below'         model has underperformed the majority baseline here
    'marginal'      at or slightly above baseline
    'adequate'      meaningfully above baseline

    Prompt A uses this to pick register. 'below' must produce hedged language
    no matter how confident today's numbers look — otherwise the report is
    confident precisely where the system has been wrong.
    """
    n = track.get("t6_n", 0)
    acc = track.get("t6_accuracy")

    if n < 15 or acc is None:
        return "insufficient"
    # Compare against THIS stock's naive baseline, not the global one. A
    # stock whose outcomes are 94% neutral sets a 94% bar; clearing 51.7%
    # there means nothing.
    base = track.get("t6_baseline")
    if base is None:
        return "insufficient"

    edge = acc - base
    if edge < 0:
        return "below"        # worse than guessing this stock's usual outcome
    if edge < 0.05:
        return "marginal"     # no meaningful edge over naive
    return "adequate"


# ═══════════════════════════════════════════════════════════════════════════
# ATTENTION ALLOCATION
#
# The scarce resource is the reader's attention, not signal. Fifteen stocks
# every day is more than anyone reads. Telling someone which two to look at
# is worth more than fifteen paragraphs of equal weight.
# ═══════════════════════════════════════════════════════════════════════════

def _attn_baseline(track):
    """Use this stock's observed baseline; gate attention if unavailable."""
    b = track.get("t6_baseline")
    return 1.0 if b is None else b


def compute_attention(archetype, t6_prob, track, t5_warning):
    """
    Returns (level, reasons). ELEVATED requires all three conditions.

    Deliberately strict. If everything is flagged, nothing is.
    """
    reasons = []

    # Guideline control (§6: automatically suppress outputs when thresholds
    # are breached). Checked FIRST: every non-quiet pattern is defined partly
    # by the model's forecast, so a stock whose forecast is withheld must not
    # earn attention from that forecast. (Before 19 Sep this check sat after
    # the NOTABLE branch, and D05/O39 were flagged NOTABLE on a withheld call.)
    _edge = track.get("t6_edge")
    if _edge is not None and _edge < SUPPRESS_EDGE:
        return "ROUTINE", [
            f"forecast withheld: model runs {abs(_edge):.0%} below this "
            f"stock's naive baseline over {track.get('t6_n', 0)} matured "
            f"forecasts"
        ]

    if archetype == "price_unavailable":
        return "ROUTINE", ["same-date price context unavailable"]
    if archetype in LOW_INFORMATION_ARCHETYPES:
        return "ROUTINE", ["no meaningful divergence"]

    if archetype not in ATTENTION_ARCHETYPES:
        return "NOTABLE", [f"{archetype}, but not a divergence pattern"]

    conds = {
        "divergence": archetype in ATTENTION_ARCHETYPES,
        "conviction": (t6_prob or 0) >= ATTENTION_MIN_T6_PROB,
        "reliable":   (track.get("t6_accuracy") or 0) >= _attn_baseline(track),
    }

    if all(conds.values()):
        reasons.append(f"{archetype} pattern")
        reasons.append(f"model probability {t6_prob:.2f}")
        reasons.append(f"{track['t6_accuracy']:.0%} accuracy vs "
                       f"{_attn_baseline(track):.0%} naive baseline on this stock")
        if t5_warning == "high":
            reasons.append("but evidence layer flagged high internal tension")
        return "ELEVATED", reasons

    failed = [k for k, v in conds.items() if not v]
    return "NOTABLE", [f"{archetype} pattern", f"missing: {', '.join(failed)}"]


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE ATTRIBUTION
#
# T5 and T6 both draw on all of Tasks 1-4. A report that talks only about
# news would misrepresent the system and mislead the reader about where the
# signal came from. Every report carries this breakdown, and Prompt A is
# required to name the top contributing sources rather than defaulting to
# news because news is the easiest thing to narrate.
# ═══════════════════════════════════════════════════════════════════════════

def attribute_sources(t5_features):
    """
    Split T5's rule weight across the four task layers.

    Returns per-source weight and share, the dominant source, and the share
    of total weight that is market-wide (identical across all 15 stocks).
    A high market-wide share means the reading is largely a market call
    wearing a single stock's name.
    """
    agg = defaultdict(lambda: {"bull": 0.0, "bear": 0.0, "n": 0,
                               "scope": "?", "factors": []})

    for f in t5_features or []:
        label, scope = SOURCE_META.get(f.get("source", "?"), ("Other", "?"))
        side = "bull" if f["direction"] == "bullish" else "bear"
        agg[label][side] += f.get("baseline_weight", 0.0)
        agg[label]["n"]  += 1
        agg[label]["scope"] = scope
        if f.get("flips_call"):
            agg[label]["factors"].append(f["factor"])

    total = sum(v["bull"] + v["bear"] for v in agg.values())
    breakdown = {}
    for label, v in agg.items():
        w = v["bull"] + v["bear"]
        breakdown[label] = {
            "weight":          round(w, 3),
            "share":           round(w / total, 3) if total else 0.0,
            "net":             round(v["bull"] - v["bear"], 3),
            "n_rules":         v["n"],
            "scope":           v["scope"],
            "decisive_factors": v["factors"],
        }

    mw = sum(v["weight"] for v in breakdown.values() if v["scope"] == "market-wide")

    return {
        "by_source":         breakdown,
        "dominant":          max(breakdown, key=lambda k: breakdown[k]["weight"])
                             if breakdown else None,
        "market_wide_share": round(mw / total, 3) if total else 0.0,
        "total_weight":      round(total, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA ADAPTATION
#
# Column names verified against the live llm_analyses table on 3 Sep 2026.
# The July monthly log planned flat consensus_ratio / model_split_count
# columns; they were never added. What exists instead is n_contradictions
# (a count) plus warning_level (the severity), which together cover what
# Task 7 needs — the full per-type breakdown lives in llm_analyses_audit's
# contradictions JSONB if it is ever needed.
#
# Discovering columns at runtime rather than hard-coding them means a future
# ALTER TABLE does not break this module.
# ═══════════════════════════════════════════════════════════════════════════

_T5_COLUMNS_CACHE = None

# Without these there is no report.
_T5_REQUIRED = ["sentiment", "confidence", "summary"]

# Used when present, silently skipped when not.
_T5_OPTIONAL = [
    "sentiment_score",      # LLM's own -1..1 strength score
    "key_drivers",          # jsonb array
    "risk_factors",         # jsonb array
    "warning_level",        # low / mild / high, from Layer 3
    "n_contradictions",     # count of detected tensions
    "baseline_sentiment",   # Layer 1 rule output before the LLM
    "baseline_confidence",  # Layer 1 confidence before the LLM
    "is_valid",             # did the six guardrails pass
    "validation_notes",
    "model_used",           # which LLM produced this row
    "source_count",
    "data_freshness",
]


def _t5_available_columns(engine):
    """Which of the columns we care about actually exist on llm_analyses."""
    global _T5_COLUMNS_CACHE
    if _T5_COLUMNS_CACHE is not None:
        return _T5_COLUMNS_CACHE

    with engine.connect() as conn:
        present = {r[0] for r in conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'llm_analyses'
        """)).fetchall()}

    missing_required = [c for c in _T5_REQUIRED if c not in present]
    if missing_required:
        raise RuntimeError(
            f"llm_analyses is missing required column(s): {missing_required}"
        )

    _T5_COLUMNS_CACHE = {
        "select":  _T5_REQUIRED + [c for c in _T5_OPTIONAL if c in present],
        "missing": [c for c in _T5_OPTIONAL if c not in present],
    }

    if _T5_COLUMNS_CACHE["missing"]:
        print(f"  ℹ️  llm_analyses has no {_T5_COLUMNS_CACHE['missing']} — "
              f"those fields will be None in reports.")

    return _T5_COLUMNS_CACHE


def build_report_input(engine, ticker, report_date,
                       t5_threshold=None, tickers=None):
    """
    Gather everything Prompt A needs for one (ticker, date). Every field is
    computed here so the LLM has nothing left to decide — it only phrases.

    Returns None when T5 is missing entirely; returns a dict with
    status='incomplete' when T5 exists but T6 does not.
    """
    t5_threshold = T5_THRESHOLD_DEFAULT if t5_threshold is None else t5_threshold
    cols = _t5_available_columns(engine)
    col_list = ", ".join(cols["select"])

    with engine.connect() as conn:
        t5_row = conn.execute(text(f"""
            SELECT {col_list}
            FROM llm_analyses
            WHERE ticker = :t AND analysis_date::date = CAST(:d AS DATE)
        """), {"t": ticker, "d": str(report_date)}).mappings().fetchone()

        t6 = conn.execute(text("""
            SELECT predicted_class, prob_bullish, prob_neutral, prob_bearish,
                   confidence, model_used, actual_return_3d, actual_class,
                   forecast_target
            FROM task6_predictions
            WHERE ticker = :t AND prediction_date = CAST(:d AS DATE)
              AND is_champion = TRUE
            ORDER BY id DESC LIMIT 1
        """), {"t": ticker, "d": str(report_date)}).fetchone()

        champ = conn.execute(text("""
            SELECT new_champion, changed_at::date
            FROM task6_champion_log
            WHERE changed_at::date <= CAST(:d AS DATE)
            ORDER BY changed_at DESC LIMIT 1
        """), {"d": str(report_date)}).fetchone()

    if t5_row is None:
        return None

    t5 = dict(t5_row)

    def _num(key):
        v = t5.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    if t6 is None:
        return {
            "ticker": ticker, "report_date": str(report_date),
            "status": "incomplete", "missing": "task6_prediction",
            "t5": {"sentiment": t5.get("sentiment"),
                   "confidence": _num("confidence") or 0.0,
                   "summary": t5.get("summary"),
                   "warning_level": t5.get("warning_level")},
        }

    t6_class  = t6[0]
    probs     = {"bullish": t6[1], "neutral": t6[2], "bearish": t6[3]}
    t6_prob   = probs.get(t6_class) or float(t6[4] or 0)
    # A historical report must not reveal this date's future three-day label.
    actual_r = t6[6] if str(t6[8]) <= str(report_date) else None
    # actual_return_3d is a THREE-DAY return, so it must be classified with
    # Task 6's three-day threshold (1.09%), not T5's next-day threshold
    # (0.604%). Using the smaller one reads ordinary drift as a directional
    # move and distorts the archetype distribution.
    #
    # LIMITATION: T5's reading is next-day-oriented (its Ridge weights are
    # fitted on next-day returns) but the only realised return available is
    # three-day. Confirmation is therefore checked across a horizon mismatch.
    # Resolving it needs a daily-return source, which does not exist yet.
    actual_c  = (classify_actual(actual_r, T6_THRESHOLD_DEFAULT)
                 if actual_r is not None else None)

    try:
        universe = tickers or _universe(engine, report_date)
        price_ctx = compute_price_context(ticker, report_date, universe)
    except Exception as e:                                       # noqa: BLE001
        print(f"  ⚠️  {ticker}: price context failed ({e!r})")
        price_ctx = None
    same_day_class = (classify_actual(price_ctx.get("return_pct"), t5_threshold)
                      if price_ctx and price_ctx.get("as_of") == str(report_date) else None)
    archetype, axis1, axis2 = classify_archetype(t5.get("sentiment"),
                                               same_day_class, t6_class, t6_prob)
    track = compute_track_record(engine, ticker, report_date, t5_threshold=t5_threshold)
    reliab = reliability_verdict(track)
    level, reasons = compute_attention(archetype, t6_prob, track, t5.get("warning_level"))

    return {
        "ticker":       ticker,
        "report_date":  str(report_date),
        "generated_at": datetime.now(_SGT).isoformat(),   # SGT, not the host clock's UTC
        "status":       "complete",

        "t5": {
            "sentiment":           t5.get("sentiment"),
            "confidence":          _num("confidence") or 0.0,
            "sentiment_score":     _num("sentiment_score"),
            "summary":             t5.get("summary"),
            "key_drivers":         t5.get("key_drivers"),
            "risk_factors":        t5.get("risk_factors"),
            "warning_level":       t5.get("warning_level"),
            "n_contradictions":    t5.get("n_contradictions"),
            # Layer 1 output before the LLM rephrased it. Guardrails force
            # sentiment to match, so a divergence here means a fallback row.
            "baseline_sentiment":  t5.get("baseline_sentiment"),
            "baseline_confidence": _num("baseline_confidence"),
            "is_valid":            t5.get("is_valid"),
            "llm_model":           t5.get("model_used"),
            "source_count":        t5.get("source_count"),
            "used_fallback":       (t5.get("is_valid") is False),
        },
        "t6": {
            "predicted_class": t6_class,
            "probability":     t6_prob,
            "probabilities":   probs,
            "champion_model":  t6[5],
            # No log row means the date predates the first swap (8/23),
            # so this model has served since deployment.
            "champion_since":  str(champ[1]) if champ else "initial",
            "champion_swapped": champ is not None,
        },
        "actual": {
            "return_3d": float(actual_r) if actual_r is not None else None,
            "class":     actual_c,
            "matured":   actual_r is not None,
        },

        "fusion": {
            "archetype":         archetype,
            "axis_confirmation": axis1,
            "axis_forecast":     axis2,
            "headline":          ARCHETYPES.get(archetype, {}).get("headline", ""),
            "note":              ARCHETYPES.get(archetype, {}).get("note", ""),
        },
        "reliability": {**track, "verdict": reliab},
        "attention":   {"level": level, "reasons": reasons},
        "price_context": price_ctx,

        "thresholds": {
            "t5_next_day":       t5_threshold,
            "t6_three_day":      T6_THRESHOLD_DEFAULT,
            "majority_baseline": MAJORITY_BASELINE,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# PRICE-HISTORY CONTEXT  (added 19 Sep, renderer fb_v4)
#
# Research R6 tested simple price patterns on these 15 stocks over 2019-2025
# (~1,870 trading days, prices only, so the long history is available even
# though Tasks 1-4 only started in June 2026):
#   - one-day relative reversal: rank IC 0.071, t = 9.8, right in 7/7 years
#   - 60-day relative momentum (skip 5 days): IC 0.039 at 5 days, t = 2.9
# The same reversal shows up in Task 5 Part A's Ridge signs (news features
# counter-intuitive, "likely short-term reversal") and in R7's rule audit.
#
# This block turns that into per-report context: where today's move ranks,
# and what has historically followed similar days for this stock and for
# the group. Everything is computed from closing prices up to and including
# the report date; a pair (day t, day t+1) is used only if t+1 <= report
# date, so nothing here uses information from after the report date.
# Statistics are descriptive averages over many days, not forecasts.
# ═══════════════════════════════════════════════════════════════════════════

PRICE_HISTORY_START = "2019-01-01"
MOM_WINDOW, MOM_SKIP, MOM_HORIZON = 60, 5, 5
_BANDS = ["bottom fifth", "second-lowest fifth", "middle fifth",
          "second-highest fifth", "top fifth"]
_PX_CACHE = {}        # {tuple(tickers): DataFrame of adjusted closes}
_PATTERN_CACHE = {}   # {(tuple(tickers), last_date): group statistics}


def _nw_t(x, lags):
    """t-stat of a mean with Newey-West variance (overlapping horizons)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return None
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else None


def load_price_panel(tickers):
    """Adjusted daily closes for the universe since PRICE_HISTORY_START.
    Downloaded once per session; callers slice to their report date.
    Adjusted prices matter here: REIT ex-dividend days would otherwise read
    as large one-day drops."""
    key = tuple(sorted(tickers))
    if key in _PX_CACHE:
        return _PX_CACHE[key]
    import pandas as pd
    import yfinance as yf
    closes = {}
    for t in key:
        try:
            h = yf.Ticker(t).history(start=PRICE_HISTORY_START, auto_adjust=True)["Close"]
            idx = h.index.tz_localize(None) if h.index.tz is not None else h.index
            h.index = idx.normalize()
            closes[t] = h
        except Exception as e:                                   # noqa: BLE001
            print(f"  ⚠️  price history for {t} unavailable ({e!r})")
    px = pd.DataFrame(closes).sort_index()
    _PX_CACHE[key] = px
    return px


def _universe(engine, report_date):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ticker FROM llm_analyses
            WHERE analysis_date::date = CAST(:d AS DATE)
        """), {"d": str(report_date)}).fetchall()
    return sorted(r[0] for r in rows)


def _group_patterns(px, rel):
    """Group-level (all stocks pooled) statistics up to the last row of px.
    Cached per (universe, last date) so a 15-report batch computes it once."""
    key = (tuple(px.columns), str(px.index[-1].date()))
    if key in _PATTERN_CACHE:
        return _PATTERN_CACHE[key]

    # one-day relative reversal: rank IC of a ranking that favours today's
    # relative LOSERS vs tomorrow's relative return (positive = reversal;
    # same sign convention as research R6's REV1)
    nxt = rel.shift(-1)
    ic1 = []
    for d in rel.index[:-1]:
        a, b = rel.loc[d], nxt.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 10 and a[ok].nunique() > 1 and b[ok].nunique() > 1:
            ic1.append((-a[ok]).rank().corr(b[ok].rank()))

    # 60-day momentum (skipping the last 5 days) vs next 5-day relative return
    mom = px.shift(MOM_SKIP) / px.shift(MOM_WINDOW + MOM_SKIP) - 1
    fwd5 = px.shift(-MOM_HORIZON) / px - 1
    ex5 = fwd5.sub(fwd5.mean(axis=1), axis=0)
    icm = []
    for d in px.index[:-MOM_HORIZON]:
        a, b = mom.loc[d], ex5.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 10 and a[ok].nunique() > 1 and b[ok].nunique() > 1:
            icm.append(a[ok].rank().corr(b[ok].rank()))
    # A day where every stock's price is unchanged (exchange holiday that
    # Yahoo still lists) has no ranking; drop any non-finite correlation.
    ic1 = [x for x in ic1 if np.isfinite(x)]
    icm = [x for x in icm if np.isfinite(x)]

    # pooled next-day relative return by band of today's relative move,
    # bands defined on each stock's own history
    by_band = {i: [] for i in range(5)}
    for t in rel.columns:
        x = rel[t].iloc[:-1].dropna()
        y = nxt[t].reindex(x.index)
        ok = y.notna()
        x, y = x[ok], y[ok]
        if len(x) < 100:
            continue
        cuts = np.nanpercentile(x.values, [20, 40, 60, 80])
        band = np.searchsorted(cuts, x.values, side="right")
        for i in range(5):
            by_band[i].extend(y.values[band == i].tolist())
    bands = []
    for i in range(5):
        v = np.array(by_band[i])
        bands.append({"band": _BANDS[i], "n": int(len(v)),
                      "mean_next_rel_pct": round(float(v.mean()), 3) if len(v) else None,
                      "share_outperformed": round(float((v > 0).mean()), 3) if len(v) else None,
                      "t": (round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 1)
                            if len(v) > 30 else None)})

    out = {
        "history_start":     str(px.index[0].date()),
        "reversal_ic":       round(float(np.mean(ic1)), 3) if ic1 else None,
        "reversal_ic_t":     round(_nw_t(ic1, 0), 1) if len(ic1) >= 10 else None,
        "reversal_days":     len(ic1),
        "momentum_ic":       round(float(np.mean(icm)), 3) if icm else None,
        "momentum_ic_t":     round(_nw_t(icm, MOM_HORIZON - 1), 1) if len(icm) >= 10 else None,
        "momentum_days":     len(icm),
        "bands":             bands,
    }
    _PATTERN_CACHE[key] = out
    return out


def compute_price_context(ticker, report_date, tickers):
    """
    Per-report price context. Returns None if prices are unavailable.

    today            : the stock's close-to-close return, the group average,
                       the relative move, and its rank (1 = strongest)
    band             : where today's relative move falls in this stock's own
                       history of daily relative moves (fifths)
    stock_after_band : on this stock's past days in the same band, the next
                       day's relative return (mean, share > 0, n, t)
    group            : pooled statistics for all stocks (see _group_patterns)
    momentum         : this stock's 60-day return rank (skipping 5 days)
    """
    import pandas as pd
    try:
        px = load_price_panel(tickers)
    except Exception as e:                                       # noqa: BLE001
        print(f"  ⚠️  price context unavailable ({e!r})")
        return None
    if ticker not in px.columns:
        return None
    px = px[px.index <= pd.Timestamp(str(report_date))]
    if len(px) < 300:
        return None
    last = px.index[-1]
    rel = px.pct_change(fill_method=None) * 100
    grp = rel.mean(axis=1)
    rel = rel.sub(grp, axis=0)          # relative to the equal-weight group

    r_today = float(px[ticker].iloc[-1] / px[ticker].iloc[-2] - 1) * 100
    rel_today = float(rel.loc[last, ticker])
    ranks = rel.loc[last].rank(ascending=False)

    # this stock's own history of (relative move, next-day relative move)
    x = rel[ticker].iloc[:-1].dropna()
    y = rel[ticker].shift(-1).reindex(x.index)
    ok = y.notna()
    x, y = x[ok], y[ok]
    cuts = np.nanpercentile(x.values, [20, 40, 60, 80])
    b_today = int(np.searchsorted(cuts, rel_today, side="right"))
    same = y.values[np.searchsorted(cuts, x.values, side="right") == b_today]
    n_same = len(same)
    t_same = (float(same.mean() / (same.std(ddof=1) / np.sqrt(n_same)))
              if n_same > 30 and same.std(ddof=1) > 0 else None)

    mom = px.shift(MOM_SKIP) / px.shift(MOM_WINDOW + MOM_SKIP) - 1
    mom_rank = mom.loc[last].rank(ascending=False)

    g = _group_patterns(px, rel)
    return {
        "as_of":            str(last.date()),
        "n_stocks":         int(rel.loc[last].notna().sum()),
        "return_pct":       round(r_today, 2),
        "group_return_pct": round(float(grp.loc[last]), 2),
        "relative_pct":     round(rel_today, 2),
        "rank":             int(ranks[ticker]),
        "band_index":       b_today,
        "band":             _BANDS[b_today],
        "stock_after_band": {
            "n":                  int(n_same),
            "mean_next_rel_pct":  round(float(same.mean()), 2) if n_same else None,
            "share_outperformed": round(float((same > 0).mean()), 3) if n_same else None,
            "t":                  round(t_same, 1) if t_same is not None else None,
        },
        "group_after_band": g["bands"][b_today],
        "group":            g,
        "momentum": {
            "window_days":  MOM_WINDOW,
            "skip_days":    MOM_SKIP,
            "horizon_days": MOM_HORIZON,
            "return_pct":   (round(float(mom.loc[last, ticker]) * 100, 1)
                             if pd.notna(mom.loc[last, ticker]) else None),
            "rank":         int(mom_rank[ticker]) if pd.notna(mom_rank[ticker]) else None,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def archetype_distribution(engine, start_date, end_date, tickers,
                           t5_threshold=None):
    """
    Run the classifier across a window and report the distribution.

    Sanity check: quiet + dormant + move_complete should be the majority. If
    lag and contradicted are common, the thresholds are too loose and the
    system is manufacturing divergence rather than detecting it.
    """
    t5_threshold = T5_THRESHOLD_DEFAULT if t5_threshold is None else t5_threshold
    counts    = defaultdict(int)
    attention = defaultdict(int)
    skipped   = 0

    with engine.connect() as conn:
        dates = [str(r[0]) for r in conn.execute(text("""
            SELECT DISTINCT analysis_date::date AS d FROM llm_analyses
            WHERE analysis_date::date BETWEEN CAST(:s AS DATE) AND CAST(:e AS DATE)
            ORDER BY d
        """), {"s": str(start_date), "e": str(end_date)}).fetchall()]

    for d in dates:
        for tkr in tickers:
            r = build_report_input(engine, tkr, d, t5_threshold)
            if r is None:
                skipped += 1
                continue
            if r.get("status") != "complete":
                counts["incomplete"] += 1
                continue
            counts[r["fusion"]["archetype"]] += 1
            attention[r["attention"]["level"]] += 1

    total = sum(counts.values())
    print("=" * 66)
    print(f"ARCHETYPE DISTRIBUTION   {start_date} → {end_date}")
    print("=" * 66)
    print(f"  {len(dates)} dates × {len(tickers)} tickers   "
          f"{total} classified, {skipped} without T5\n")

    for k in list(ARCHETYPES) + ["incomplete"]:
        n = counts.get(k, 0)
        if not n:
            continue
        pct = n / total * 100 if total else 0
        mark = "·" if k in LOW_INFORMATION_ARCHETYPES else \
               "★" if k in ATTENTION_ARCHETYPES else " "
        print(f"  {mark} {k:<16} {n:>4}  ({pct:>5.1f}%)  {'█' * int(pct / 2)}")

    quiet = sum(counts.get(k, 0) for k in LOW_INFORMATION_ARCHETYPES)
    if total:
        qpct = quiet / total * 100
        print(f"\n  low-information total: {qpct:.1f}%")
        if qpct < 55:
            print(f"  ⚠️  Below 55% — thresholds may be too loose; the system")
            print(f"      is flagging more than it should.")
        else:
            print(f"  ✅ Most days are correctly identified as uneventful.")

    print(f"\n  Attention: {dict(attention)}")
    if total:
        elev = attention.get("ELEVATED", 0) / len(dates) if dates else 0
        print(f"  ELEVATED per day: {elev:.1f} (target 1-3 of {len(tickers)})")

    return dict(counts), dict(attention)
