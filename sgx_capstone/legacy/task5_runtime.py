# ═══════════════════════════════════════════════════════════════════════════
# TASK 5 — PART B: RUNTIME FUNCTIONS (Step 6-13)
#
# Function definitions used every day for every stock. Run once per session
# after Part A. Where Part A trains the models once, Part B uses them to
# produce one explainable analysis per stock.
#
# ───────────────────────────────────────────────────────────────────────────
# STEP OVERVIEW (each step's purpose)
# ───────────────────────────────────────────────────────────────────────────
#   Step 6   compose_stock_context        Read 4 Supabase tables into one dict.
#   Step 7   compute_feature_contributions Layer 1 — rule-based scoring using
#                                          Ridge-derived weights.
#   Step 8   compute_shap_explanation      Layer 2 — independent SHAP second
#                                          opinion (LogReg + RF).
#   Step 9   detect_contradictions         Layer 3 — flag where Layer 1 and 2,
#                                          or individual features, disagree.
#   Step 10  build_llm_prompt              Layer 4 — assemble prompt with news.
#   Step 11  call_groq_llm                 Layer 4 — LLM inference (Groq Llama).
#   Step 12  validate_llm_output           Layer 4 — enforce sentiment bounds.
#   Step 13  store_llm_analysis            Upsert result + append audit log.
#
# ───────────────────────────────────────────────────────────────────────────
# REQUIRES (from Part A)
# ───────────────────────────────────────────────────────────────────────────
#   extract_features_row   feature extractor (single source of truth — the
#                          SAME function used in Part A training, no drift)
#   DATA_DRIVEN_WEIGHTS    Ridge-derived weights
#   DATA_DRIVEN_SIGNS      Ridge coefficient signs
#   DATA_DRIVEN_STABILITY  CV sign consistency
#   TASK5_ARTIFACTS        SHAP models + explainers + scaler + thresholds
#   FEATURE_LABEL_MAP      human-readable feature labels
#   BULLISH_EVENTS, BEARISH_EVENTS
#
# REQUIRES (from Setup): SUPABASE_ENGINE, SGX_STOCKS, GROQ_API_KEY
#
# ───────────────────────────────────────────────────────────────────────────
# DESIGN PRINCIPLE
# ───────────────────────────────────────────────────────────────────────────
#   The LLM is a WRITER, not a decision-maker. It may not reverse the
#   rule-based sentiment. Authority comes from Ridge weights + SHAP models;
#   the LLM only synthesizes them into natural language. When Layer 1 and
#   Layer 2 disagree, that disagreement is surfaced (Layer 3), not hidden.
#
#   (Full fix history + backlog is in the external doc, not inlined here.)
# ═══════════════════════════════════════════════════════════════════════════

import json
import hashlib
import requests
import time
import pickle
import os
import numpy as np
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

# Groq model — Llama 3.1 8B (fast, free tier: 30 req/min)
GROQ_MODEL = CFG["llm"]["model"]

# Singapore timezone (used by Part D for correct dates)
SGT = timezone(timedelta(hours=8))

# ── Which Part A model the runtime uses [22 Sep] ────────────────────────────
# Set this after the v6-vs-v7 hold-out comparison cell: it prints ADOPT v7 or
# KEEP v6. Part B always loads THIS file from Drive, never whatever Part A
# happens to have left in memory, so running a Part A training cell cannot
# silently change the model the daily runner uses.
T5_MODEL_VERSION = "v6"          # "v6" or "v7"
ARTIFACTS_PATH = globals().get("ARTIFACTS_PATH", f"/app/models/task5_artifacts_{T5_MODEL_VERSION}.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# ARTIFACT RELOAD GUARD [DD5, extended v6]
#
# If Part A's globals were cleared by a kernel restart, reload everything
# from Drive instead of forcing a full retrain.
#
# [FIX v6] The original guard restored only TASK5_ARTIFACTS. But Part B also
# needs DATA_DRIVEN_WEIGHTS / SIGNS / STABILITY, which Part A built in memory
# and never persisted. So after a restart Part B would load "successfully",
# then throw NameError at the first _get_weight() call. The bug never surfaced
# in July because Part A and Part B were always run back-to-back in the same
# session. Part A v6 now saves those tables; this guard restores them.
#
# Restoring from the pickle also means a Part A retrain that changes the
# feature set needs no edits here — v5 had 19 features, v6 has 20
# (regime_encoded returned, options_bearish was dropped as collinear).
# ═══════════════════════════════════════════════════════════════════════════

# [22 Sep] Always load the pinned version from Drive. Before, this only
# loaded when TASK5_ARTIFACTS was absent, so a Part A run in the same session
# decided the model implicitly.
if not os.path.exists(ARTIFACTS_PATH):
    raise RuntimeError(f"{ARTIFACTS_PATH} not found. Run the Part A cell for "
                       f"{T5_MODEL_VERSION} first, or set T5_MODEL_VERSION.")
with open(ARTIFACTS_PATH, "rb") as f:
    TASK5_ARTIFACTS = pickle.load(f)
if TASK5_ARTIFACTS.get("model_version") != T5_MODEL_VERSION:
    raise RuntimeError(f"{ARTIFACTS_PATH} contains model_version "
                       f"{TASK5_ARTIFACTS.get('model_version')!r}, expected "
                       f"{T5_MODEL_VERSION!r}.")
print(f"✅ Loaded TASK5_ARTIFACTS {T5_MODEL_VERSION} from Drive")


def _load_rule_weights(artifacts):
    """
    Restore the rule-layer weight tables from the Part A pickle.

    Returns (weights, signs, stability, feature_names, source_description).

    Sign convention matches Part A exactly: numeric np.sign() output
    (-1.0 / 0.0 / +1.0), not "+"/"-" strings. Part B compares these
    numerically, so the convention must not drift.
    """
    feature_names = list(artifacts["feature_names"])

    if "data_driven_weights" in artifacts:
        weights   = dict(artifacts["data_driven_weights"])
        signs     = dict(artifacts.get("data_driven_signs", {}))
        stability = dict(artifacts.get("data_driven_stability", {}))
        source    = "pickle"

    elif "ridge_coefs" in artifacts:
        # Fallback: recompute the same normalisation Part A applies
        # (scaled_weight = |coef| / max|coef|, Part A Step 4).
        coefs     = {k: float(v) for k, v in dict(artifacts["ridge_coefs"]).items()}
        max_abs   = max(abs(v) for v in coefs.values()) or 1.0
        weights   = {k: abs(v) / max_abs for k, v in coefs.items()}
        signs     = {k: float(np.sign(v)) for k, v in coefs.items()}
        stability = dict(artifacts.get("data_driven_stability", {}))
        source    = "pickle (derived from ridge_coefs)"

    else:
        raise KeyError(
            "TASK5_ARTIFACTS contains no weight data.\n"
            f"Available keys: {sorted(artifacts.keys())}\n\n"
            "Re-run Part A v6, which saves data_driven_weights / signs / "
            "stability / ridge_coefs into the pickle."
        )

    # Fill gaps so downstream lookups never KeyError
    for feat in feature_names:
        weights.setdefault(feat, 0.1)
        signs.setdefault(feat, 0.0)
        stability.setdefault(feat, 1.0)

    return weights, signs, stability, feature_names, source


# Always from the pinned pickle (see T5_MODEL_VERSION above).
(DATA_DRIVEN_WEIGHTS, DATA_DRIVEN_SIGNS,
 DATA_DRIVEN_STABILITY, FEATURE_NAMES, _wsrc) = _load_rule_weights(TASK5_ARTIFACTS)
print(f"✅ Loaded rule weights from {_wsrc}")

print(f"   version:  {TASK5_ARTIFACTS.get('model_version', '?')}  "
      f"(trained {TASK5_ARTIFACTS.get('train_window', '?')})")
print(f"   features: {len(FEATURE_NAMES)}")


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE-SET DRIFT CHECK [v6]
#
# Rules reference features by name. When Part A's Step 3.7 drops a collinear
# or zero-variance column, any rule referencing it silently falls back to
# _get_weight()'s default of 0.1 — the rule still fires, but with a magnitude
# unrelated to the data. Surface that at load time rather than letting it run
# unnoticed in production.
# ═══════════════════════════════════════════════════════════════════════════

_RULE_REFERENCED_FEATURES = {
    "news_bullish_ratio", "news_bearish_ratio", "avg_sentiment", "n_articles",
    "n_political", "political_direction", "n_bullish_events", "n_bearish_events",
    "rsi", "macd_bullish_crossover", "macd_bearish_crossover",
    "bb_above_upper", "bb_below_lower", "vwap_above", "weekly_bullish",
    "regime_encoded", "vix", "yield_spread", "options_bullish",
    "options_bearish", "pc_vol_ratio",
}

_missing_feats = sorted(_RULE_REFERENCED_FEATURES - set(FEATURE_NAMES))
_unused_feats  = sorted(set(FEATURE_NAMES) - _RULE_REFERENCED_FEATURES)

if _missing_feats:
    print(f"\n  ⚠️  {len(_missing_feats)} rule feature(s) not in this model — "
          f"those rules fire with fallback weight 0.1:")
    for _f in _missing_feats:
        print(f"       {_f}")
if _unused_feats:
    print(f"\n  ℹ️  {len(_unused_feats)} model feature(s) no rule uses "
          f"(their Ridge signal never reaches the rule layer): {_unused_feats}")
if not _missing_feats and not _unused_feats:
    print("  ✅ Rule feature set matches model feature set exactly.")


# ── regime_encoded fallback ─────────────────────────────────────────────────
# Part A one-hots macro_regime, then Step 3.7 may drop the encoded column as
# zero variance when every training day was Neutral. The macro rules below
# still reference it, so give it a small fixed weight to prevent the fallback
# warning. These rules only fire on Recessionary / Risk-Off / Risk-On days.
#
# In v6 regime_encoded survived Step 3.7 with coefficient +0.1684 (third
# largest), so setdefault leaves the real weight untouched.
DATA_DRIVEN_WEIGHTS.setdefault("regime_encoded", 0.15)

# ═══════════════════════════════════════════════════════════════════════════
# TUNABLE CONSTANTS
#
# Everything below is a judgment call. Each has a comment stating the value,
# the reason, and its known limitation. they are design parameters made explicit
# so they can be revised.
# ═══════════════════════════════════════════════════════════════════════════

# ── Rule-layer version ──────────────────────────────────────────────────────
# Independent of TASK5_ARTIFACTS["model_version"]. The rule layer can change
# without a retrain, and did: v7 added regime_encoded to MW1 on 3 September
# against the unchanged v6 model. Every stored row carries this so a future
# reader can tell which rules produced it.
#
#   rl_v6  MW1 = {n_political, political_direction, vix, yield_spread,
#                 pc_vol_ratio}. regime_encoded returned from Part A at
#                 weight 0.853 and ran undiscounted. Produced uniformly
#                 bullish output on 17-18 August.
#   rl_v7  regime_encoded added to MW1. Coverage assertion added.
#   rl_v8  [EV1] corporate-event rules fire only on ABNORMAL event activity
#          (19 Sep). Research R7 showed the bullish event rule fired on 99.7%
#          of rl_v7 readings and supplied +0.203 of the +0.381 average net
#          score: Task 1 keyword-tags ~35 articles per stock per day (9.2
#          bullish-event articles on average), so "any earnings/dividend
#          headline today" is true almost every day. R8 replay (348 readings,
#          criteria fixed in advance): firing 99.7% → 26.1%, mean net
#          +0.381 → +0.241, next-day hit vs peers and rank IC unchanged.
RULE_LAYER_VERSION = "rl_v8"

BASE_CONFIDENCE = TASK5_ARTIFACTS.get("base_confidence", 0.55)   # from Part A [DD2]
MAX_CONFIDENCE  = 0.85   # signal-strength ceiling (this is NOT a probability;
                         # calibrated probability belongs to Task 6)

# ── Next-day label threshold [v6] ───────────────────────────────────────────
# Part A recomputes this on the final post-filter sample. v6 landed at
# ±0.604% on 585 samples. Stored in the pickle as "return_threshold".
#
# Task 7 must use THIS value when checking whether a T5 reading was confirmed
# by the next-day move. Task 6 uses ±1.09% because it labels 3-day returns —
# a different volatility scale. Applying Task 6's threshold to T5 would
# misclassify many confirmed days as "no reaction" and skew the archetype
# distribution toward the muted row.
T5_LABEL_THRESHOLD = float(TASK5_ARTIFACTS.get("return_threshold", 0.604))

# ── Sentiment threshold [MW3] ───────────────────────────────────────────────
# Net-score cutoff for the bullish / neutral / bearish call. Tuned so quiet
# days sit at neutral rather than being pushed into a direction by small
# residual signal. Judgment call.
NET_SCORE_THRESHOLD = 0.35

# ── Market-wide feature discount [MW1] ──────────────────────────────────────
# These features take the SAME value for all 15 stocks on a given day, so a
# single value moves every stock the same way. Discounted to 0.25 so they
# tilt the reading without deciding it.
#
# ORIGINAL JUSTIFICATION (v5, July, 375 samples / 43 days)
#   Step 4.5's temporal check showed every macro / options coefficient FLIPPED
#   sign out of sample (0 of 4 held), while news features mostly held (6 of 8).
#   With 43 days a market-wide feature has only 43 distinct values regardless
#   of row count, so its Ridge magnitude was fitted on too few effective
#   observations to trust at full strength.
#
# REVISED FINDING (v6, September, 585 samples / 47 days)
#   The statistical argument weakened. With 56% more samples, macro and options
#   coefficients held 4 of 4 under the same date-split check, and OOS R² turned
#   positive (-0.190 → +0.008). Sign instability was a sample-size artefact.
#
#   Technical features moved the other way, holding only 3 of 7 — both MACD
#   crossovers flipped, suggesting momentum reversed within the 10-day test
#   window. Ten days is too short to separate a regime effect from a
#   persistent one, so that is not yet grounds for a new discount rule.
#
# WHY THE DISCOUNT STAYS
#   The product argument is unaffected by sample size. Identical values across
#   all 15 stocks destroy the cross-sectional variety the system needs to say
#   anything distinct about individual stocks.
#
# [v7] regime_encoded ADDED — measured, not assumed
#   Part A v6 brought regime_encoded back (it was zero-variance in v5 and
#   dropped). It returned as the THIRD largest coefficient, +0.1684, weight
#   0.853. It was not in this set, so it fired at full strength on every stock
#   with no counterweight.
#
#   The bank-vs-REIT diagnostic over 17-19 August measured the consequence:
#
#     date   macro rule   shared net   outcome across 15 stocks
#     8/17   FIRED           +0.861    15 bullish,  0 neutral
#     8/18   FIRED           +1.003    15 bullish,  0 neutral
#     8/19   absent          -0.092     3 bullish,  9 neutral
#
#   The rule fired on exactly the two days that came back uniformly bullish and
#   was absent on the one day stocks differentiated. Source attribution on 8/17
#   put it at 45.3% of all rule weight, 12.79 bullish against 0.00 bearish —
#   +0.853 added to all 15 stocks, unopposed, against a decision threshold of
#   0.35. One rule at 2.4x the threshold.
#
#   Contrast political, which carries 26.1% of weight but nets only +0.12
#   because MW5's counterweight cancels it. That is MW1 + MW5 working. Regime
#   had neither.
#
#   This is the same structural fault as July's all-bullish bug, recurring on a
#   different feature: a market-wide feature entering the rule layer at full
#   weight with nothing opposing it. The general lesson, recorded here for the
#   final report, is that any feature sharing one value across all stocks must
#   be checked for cross-sectional dominance before it is trusted at full
#   weight. The assertion below enforces that going forward.
#
#   Projected effect at 0.25: weight 0.853 → 0.213, removing 0.640 per stock.
#   On 8/17 D05 goes +1.035 → +0.395 (still bullish) while ME8U goes
#   +0.794 → +0.154 (neutral). Stocks differentiate. Market-wide share of
#   total weight falls from 71.4% to roughly 57%.
MARKET_WIDE_FEATURES = {
    "n_political", "political_direction",
    "vix", "yield_spread", "pc_vol_ratio",
    "regime_encoded",          # [v7] see diagnostic above
}
MARKET_WIDE_SCALE = 0.25

# ── Market-wide coverage assertion [v7] ─────────────────────────────────────
# Part A's Step 3.7 drops and restores features as the sample grows, so the
# feature set is not stable across retrains. regime_encoded was absent in v5
# and third-largest in v6, and Part B had no way to notice. This check runs at
# load time: any feature sourced from a market-level table that is not covered
# by MW1 gets flagged before it can quietly dominate a backfill.
_MARKET_WIDE_CANDIDATES = {
    "vix", "yield_spread", "pc_vol_ratio",
    "options_bullish", "options_bearish", "regime_encoded",
    "n_political", "political_direction",
}
_uncovered_mw = sorted(
    f for f in _MARKET_WIDE_CANDIDATES
    if f in FEATURE_NAMES and f not in MARKET_WIDE_FEATURES
)
if _uncovered_mw:
    print(f"\n  ⚠️  MARKET-WIDE FEATURE AT FULL WEIGHT: {_uncovered_mw}")
    print(f"      Same value for all 15 stocks, no MW1 discount. Check for")
    print(f"      cross-sectional dominance before running a backfill.")
    for _f in _uncovered_mw:
        _w = DATA_DRIVEN_WEIGHTS.get(_f, 0)
        _flag = "  ← exceeds decision threshold on its own" \
                if _w > NET_SCORE_THRESHOLD else "  (below threshold, cannot force a call alone)"
        print(f"        {_f:<24} weight {_w:.3f}{_flag}")
    print(f"      Threshold for reference: {NET_SCORE_THRESHOLD}")
else:
    print(f"  ✅ All market-wide features covered by MW1 (scale {MARKET_WIDE_SCALE})")

# KNOWN AND ACCEPTED: options_bullish will trip this on v6.
#
# It is market-wide (Task 4 uses EWS as a single market proxy, one row per day)
# and not in MARKET_WIDE_FEATURES. Left uncovered deliberately: its v6 weight is
# 0.136, well under the 0.35 threshold, so it cannot force a call the way
# regime_encoded's 0.853 did. The 8/17 attribution measured options at 0.0% of
# total weight — it did not fire at all.
#
# Not suppressing the warning is the point. A future retrain could push it up,
# and the warning prints its current weight next to the threshold so the
# decision can be re-made against real numbers rather than this comment.

# ── Event weight discount [MW2] ─────────────────────────────────────────────
# Corporate events are classified by TYPE (ma, earnings, dividend, upgrade),
# but type doesn't reliably imply direction — a REIT "issues units" is ma
# but dilutive, an earnings release can be a miss, a dividend can be a cut.
# Since the DIRECTION is unreliable, event weights are discounted so they
# nudge rather than dominate.
EVENT_SCALE = 0.5

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# Track which features we've warned about (warn once per feature, avoid spam)
_MISSING_FEATURES_WARNED = set()

_MISSING_FEATURES_WARNED = set()

def _get_weight(feature_key, fallback=0.1):
    """
    Ridge magnitude for one feature. This is a MAGNITUDE, not a signed weight
    — direction is decided by the rule that references the feature (domain
    logic), because several Ridge signs are stably counter-intuitive
    (documented in Part A as short-term reversal).

    [MW1] Market-wide features are discounted here; the reason is in the
          MARKET_WIDE_SCALE comment.
    """
    if feature_key in DATA_DRIVEN_WEIGHTS:
        w = DATA_DRIVEN_WEIGHTS[feature_key]
    else:
        if feature_key not in _MISSING_FEATURES_WARNED:
            print(f"    ⚠️  Feature '{feature_key}' not in DATA_DRIVEN_WEIGHTS, "
                  f"using fallback {fallback}.")
            _MISSING_FEATURES_WARNED.add(feature_key)
        w = fallback

    if feature_key in MARKET_WIDE_FEATURES:
        w *= MARKET_WIDE_SCALE
    return w

# ═══════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED CONSTANTS [FIX v6]
#
# These were previously defined in Part A or in a separate setup cell, which
# meant Part B silently depended on cell execution order. After a kernel
# restart the reload guard restored TASK5_ARTIFACTS but not these, so Part B
# loaded fine and then threw NameError deep inside the pipeline —
# FEATURE_LABEL_MAP inside compute_shap_explanation, BULLISH_EVENTS inside
# compute_feature_contributions.
#
# Defining them here makes Part B runnable from a cold kernel with only the
# pickle plus SUPABASE_ENGINE / SGX_STOCKS / GROQ_API_KEY from Setup.
# ═══════════════════════════════════════════════════════════════════════════

# ── Human-readable feature labels for the LLM prompt ────────────────────────
# The LLM sees these instead of raw column names. Anything not listed falls
# through unchanged, so a new feature degrades to its column name rather
# than crashing.
FEATURE_LABEL_MAP = {
    # Task 1 — news and sentiment
    "news_bullish_ratio":     "positive news share",
    "news_bearish_ratio":     "negative news share",
    "avg_sentiment":          "average news sentiment",
    "n_articles":             "news coverage volume",
    "n_political":            "political news volume",
    "political_direction":    "political news tone",
    "n_bullish_events":       "positive corporate events",
    "n_bearish_events":       "negative corporate events",

    # Task 2 — technical
    "rsi":                    "RSI (momentum)",
    "macd_bullish_crossover": "MACD bullish crossover",
    "macd_bearish_crossover": "MACD bearish crossover",
    "bb_above_upper":         "price above upper Bollinger band",
    "bb_below_lower":         "price below lower Bollinger band",
    "vwap_above":             "price above VWAP",
    "weekly_bullish":         "weekly trend confirmation",

    # Task 3 — macro
    "regime_encoded":         "macro regime",
    "vix":                    "VIX (volatility index)",
    "yield_spread":           "yield curve spread",

    # Task 4 — options
    "options_bullish":        "bullish options positioning",
    "options_bearish":        "bearish options positioning",
    "pc_vol_ratio":           "put/call volume ratio",
}

# ── Corporate event direction ───────────────────────────────────────────────
# Type does not reliably imply direction — a REIT issuing units is tagged "ma"
# but is dilutive, an earnings release can be a miss, a dividend can be a cut.
# That unreliability is why [MW2] discounts event weights rather than treating
# them as strong signals.
BULLISH_EVENTS = {"upgrade", "earnings", "dividend", "ma"}   # ma = merger/acquisition
BEARISH_EVENTS = {"warning", "downgrade", "layoff"}

# ── Abnormal-event gate [EV1, rl_v8] ────────────────────────────────────────
# An event rule fires only when today's number of event-tagged articles for
# the stock exceeds its own average over its previous EVENT_TRAIL_DAYS news
# days by at least EVENT_EXCESS_MIN. Same idea as [MW5], which compares
# n_political with its training mean: a count only carries information
# relative to its usual level. With fewer than EVENT_MIN_HISTORY prior days
# the rule does not fire (no baseline to compare with).
# Weights, directions and [MW2]'s 50% discount are unchanged.
EVENT_TRAIL_DAYS  = 20
EVENT_EXCESS_MIN  = 1.0
EVENT_MIN_HISTORY = 5


def _map_to_readable(feature_name):
    """Convert internal feature name → human-readable label for the LLM prompt."""
    return FEATURE_LABEL_MAP.get(feature_name, feature_name)

def _match_sector(sector, keywords):
    """
    Robust sector matching [B1]. SGX_STOCKS uses variants like "Banking",
    "Financials", "REIT", "Real Estate", "Property", "Trust". Case-insensitive
    substring match so sector rules fire regardless of exact label.
    """
    sector_lower = (sector or "").lower()
    return any(k in sector_lower for k in keywords)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: compose_stock_context
#
# Purpose: assemble all data for one (ticker, date) from 4 Supabase tables into
# one dict. Includes the top 10 news headlines WITH content preview so the LLM
# can cite specific events ("DBS Q2 profit +15%") rather than generic phrasing.
# ═══════════════════════════════════════════════════════════════════════════

def compose_stock_context(ticker, date, engine=None):
    """
    Build complete data context for one stock on one date.

    Reads: news_articles (Task 1, ticker + POLITICAL), technical_signals
    (Task 2), macro_indicators (Task 3, market-wide), market_options (Task 4,
    market-wide). Returns a dict with keys: ticker, name, sector, date, news,
    political, technical, macro, options.
    """
    if engine is None:
        engine = SUPABASE_ENGINE

    context = {
        "ticker": ticker,
        "name":   SGX_STOCKS[ticker][0],
        "sector": SGX_STOCKS[ticker][1],
        "date":   date,
    }

    # Single connection for all queries (avoids reconnect overhead)
    with engine.connect() as conn:
        # ── 1. Company news aggregates ──
        news_row = conn.execute(text("""
            SELECT
                COUNT(*)                                              AS total,
                SUM(CASE WHEN sentiment='bullish' THEN 1 ELSE 0 END)  AS bull,
                SUM(CASE WHEN sentiment='neutral' THEN 1 ELSE 0 END)  AS neut,
                SUM(CASE WHEN sentiment='bearish' THEN 1 ELSE 0 END)  AS bear,
                AVG(sentiment_score)                                  AS avg_score,
                MAX(low_coverage)                                     AS low_cov
            FROM news_articles
            WHERE ticker = :t AND date = :d AND category = 'company'
        """), {"t": ticker, "d": date}).fetchone()

        # Top 10 titles by |sentiment_score| (most impactful first), with content
        top_titles = conn.execute(text("""
            SELECT title, sentiment, sentiment_score, content
            FROM news_articles
            WHERE ticker = :t AND date = :d AND category = 'company'
            ORDER BY ABS(COALESCE(sentiment_score, 0)) DESC
            LIMIT 10
        """), {"t": ticker, "d": date}).fetchall()

        # Distinct events that fired today (earnings, dividend, upgrade, etc.)
        events = conn.execute(text("""
            SELECT DISTINCT event_flag
            FROM news_articles
            WHERE ticker = :t AND date = :d
              AND category = 'company' AND event_flag != 'none'
        """), {"t": ticker, "d": date}).fetchall()

        # [EV1] Today's event-tagged article counts vs this stock's own
        # trailing average (previous EVENT_TRAIL_DAYS news days, today excluded).
        evt_today = conn.execute(text("""
            SELECT
                SUM(CASE WHEN event_flag IN ('upgrade','earnings','dividend','ma')
                         THEN 1 ELSE 0 END),
                SUM(CASE WHEN event_flag IN ('warning','downgrade','layoff')
                         THEN 1 ELSE 0 END)
            FROM news_articles
            WHERE ticker = :t AND date = :d AND category = 'company'
        """), {"t": ticker, "d": date}).fetchone()
        evt_trail = conn.execute(text("""
            WITH daily AS (
                SELECT date,
                       SUM(CASE WHEN event_flag IN ('upgrade','earnings','dividend','ma')
                                THEN 1 ELSE 0 END) AS nb,
                       SUM(CASE WHEN event_flag IN ('warning','downgrade','layoff')
                                THEN 1 ELSE 0 END) AS nr
                FROM news_articles
                WHERE ticker = :t AND date < :d AND category = 'company'
                GROUP BY date
                ORDER BY date DESC
                LIMIT :n
            )
            SELECT AVG(nb), AVG(nr), COUNT(*) FROM daily
        """), {"t": ticker, "d": date, "n": EVENT_TRAIL_DAYS}).fetchone()

        context["news"] = {
            "company_articles": news_row[0] or 0,
            "bullish_count":    news_row[1] or 0,
            "neutral_count":    news_row[2] or 0,
            "bearish_count":    news_row[3] or 0,
            "avg_sentiment":    round(news_row[4], 3) if news_row[4] else 0.0,
            "low_coverage":     bool(news_row[5]) if news_row[5] is not None else False,
            "events":           [e[0] for e in events],
            "event_counts": {
                "bull_today":   float(evt_today[0] or 0),
                "bear_today":   float(evt_today[1] or 0),
                "bull_trail":   float(evt_trail[0]) if evt_trail[0] is not None else None,
                "bear_trail":   float(evt_trail[1]) if evt_trail[1] is not None else None,
                "trail_days":   int(evt_trail[2] or 0),
            },
            "top_titles":       [
                {
                    "title":     t[0],
                    "sentiment": t[1],
                    "score":     round(t[2], 3) if t[2] else 0,
                    "content":   (t[3] or ""),   # full content (see build_llm_prompt B3)
                }
                for t in top_titles
            ],
        }

        # ── 2. Political news (market-wide, shared across all stocks) ──
        pol_row = conn.execute(text("""
            SELECT
                COUNT(*)                                              AS total,
                SUM(CASE WHEN sentiment='bullish' THEN 1 ELSE 0 END)  AS bull,
                SUM(CASE WHEN sentiment='bearish' THEN 1 ELSE 0 END)  AS bear,
                AVG(sentiment_score)                                  AS avg_score
            FROM news_articles
            WHERE ticker='POLITICAL' AND date=:d AND category='political'
        """), {"d": date}).fetchone()

        pol_titles = conn.execute(text("""
            SELECT title, sentiment, sentiment_score, content
            FROM news_articles
            WHERE ticker='POLITICAL' AND date=:d
            ORDER BY ABS(COALESCE(sentiment_score, 0)) DESC
            LIMIT 3
        """), {"d": date}).fetchall()

        context["political"] = {
            "total":          pol_row[0] or 0,
            "bullish_count":  pol_row[1] or 0,
            "bearish_count":  pol_row[2] or 0,
            # avg_sentiment == political_direction feature (used by Layer 1 [DD3])
            "avg_sentiment":  round(pol_row[3], 3) if pol_row[3] else 0.0,
            "top_titles":     [
                {
                    "title":     t[0],
                    "sentiment": t[1],
                    "score":     round(t[2], 3) if t[2] else 0,
                    "content":   (t[3] or ""),
                }
                for t in pol_titles
            ],
        }

        # ── 3. Technical signals (ticker-specific) ──
        tech_row = conn.execute(text("""
            SELECT rsi, rsi_signal, macd_value, macd_signal, bb_position,
                   vwap_position, atr, volume_signal, overall_signal, weekly_signal
            FROM technical_signals
            WHERE ticker = :t AND date = :d
        """), {"t": ticker, "d": date}).fetchone()

        context["technical"] = ({
            "rsi":            tech_row[0],
            "rsi_signal":     tech_row[1],
            "macd_value":     tech_row[2],
            "macd_signal":    tech_row[3],
            "bb_position":    tech_row[4],
            "vwap_position":  tech_row[5],
            "atr":            tech_row[6],
            "volume_signal":  tech_row[7],
            "overall_signal": tech_row[8],
            "weekly_signal":  tech_row[9],
        } if tech_row else None)

        # ── 4. Macro indicators (market-wide) ──
        macro_row = conn.execute(text("""
            SELECT sg_interest_rate_pct, sg_cpi_yoy_pct, usd_sgd_rate, vix,
                   us_yield_spread, macro_regime, sector_impact
            FROM macro_indicators
            WHERE date = :d
        """), {"d": date}).fetchone()

        context["macro"] = ({
            "regime":        macro_row[5],
            "sora":          macro_row[0],
            "cpi_yoy":       macro_row[1],
            "usd_sgd":       macro_row[2],
            "vix":           macro_row[3],
            "yield_spread":  macro_row[4],
            "sector_impact": macro_row[6],
        } if macro_row else None)

        # ── 5. Options signal (market-wide) ──
        opt_row = conn.execute(text("""
            SELECT options_signal, pc_volume_ratio, pc_oi_ratio,
                   unusual_count, ref_pc_ratio
            FROM market_options
            WHERE date = :d
        """), {"d": date}).fetchone()

        context["options"] = ({
            "signal":          opt_row[0],
            "pc_volume_ratio": opt_row[1],
            "pc_oi_ratio":     opt_row[2],
            "unusual_count":   opt_row[3],
            "spy_pc_ratio":    opt_row[4],
        } if opt_row else None)

    return context

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: compute_feature_contributions  (Layer 1 — rule-based)
#
# Purpose: score sentiment by summing Ridge-derived weights of features that
# "fire". Each weight has statistical backing (Ridge on next-day returns)
# rather than being hand-picked.
#
# Thresholds/multipliers that are NOT data-driven are documented inline as
# expert priors: news ±0.15 (Tetlock 2007), RSI 70/30 (Wilder 1978), and the
# regime "gravity" multipliers (×2.0 recessionary, ×1.5 risk-off, ×1.2 sector)
# which linear Ridge cannot capture.
# ═══════════════════════════════════════════════════════════════════════════

def compute_feature_contributions(context):
    """
    Compute rule-based sentiment + per-feature contributions.

    Returns dict with:
      features            triggered features (direction + weight + evidence)
      baseline_sentiment  bullish / neutral / bearish
      baseline_confidence signal strength (see [DD2] — not calibrated probability)
      score_breakdown     raw bullish/bearish/net scores for auditing
    """
    features = []
    bullish_score = 0.0
    bearish_score = 0.0

    news    = context.get("news", {})
    n_bull  = news.get("bullish_count", 0)
    n_neut  = news.get("neutral_count", 0)
    n_bear  = news.get("bearish_count", 0)
    n_total = n_bull + n_neut + n_bear
    events  = news.get("events", [])

    # ── News direction ──
    # d_score = (bull - bear) / total. Threshold ±0.15 (Tetlock 2007).
    if n_total > 0:
        d_score = (n_bull - n_bear) / n_total
        w_bull  = _get_weight("news_bullish_ratio")
        w_bear  = _get_weight("news_bearish_ratio")

        if d_score > 0.15:
            features.append({
                "factor":          "Positive company news sentiment",
                "source":          "task1_news",
                "direction":       "bullish",
                "baseline_weight": round(w_bull, 3),
                "raw_value":       round(d_score, 3),
                "evidence": [
                    f"{n_bull} bullish, {n_neut} neutral, {n_bear} bearish out of {n_total} articles",
                    f"Direction ratio: +{d_score:.2f} (threshold: +0.15, per Tetlock 2007)"
                ]
            })
            bullish_score += w_bull
        elif d_score < -0.15:
            features.append({
                "factor":          "Negative company news sentiment",
                "source":          "task1_news",
                "direction":       "bearish",
                "baseline_weight": round(w_bear, 3),
                "raw_value":       round(d_score, 3),
                "evidence": [
                    f"{n_bull} bullish, {n_neut} neutral, {n_bear} bearish out of {n_total} articles",
                    f"Direction ratio: {d_score:.2f} (threshold: -0.15)"
                ]
            })
            bearish_score += w_bear

    # ── Corporate events [DD4] ──
    # Uses count-based n_bullish_events / n_bearish_events weights (matching
    # Part A v4), not the old binary has_*_events weights.
    # [MW2] Weights discounted by EVENT_SCALE — event TYPE doesn't reliably
    # imply direction (a REIT "issues units" is dilutive, "earnings" can miss).
    # [EV1, rl_v8] Fire only on abnormal event activity for this stock.
    ec = news.get("event_counts") or {}

    def _abnormal(side):
        today, trail = ec.get(f"{side}_today"), ec.get(f"{side}_trail")
        if today is None or trail is None or ec.get("trail_days", 0) < EVENT_MIN_HISTORY:
            return False, None
        return (today - trail) >= EVENT_EXCESS_MIN, today - trail

    if events:
        bull_evt = [e for e in events if e in BULLISH_EVENTS]
        bear_evt = [e for e in events if e in BEARISH_EVENTS]
        bull_ok, bull_x = _abnormal("bull")
        bear_ok, bear_x = _abnormal("bear")

        if bull_evt and bull_ok:
            w = _get_weight("n_bullish_events") * EVENT_SCALE   # [MW2]
            features.append({
                "factor":          f"Unusual bullish corporate event activity ({len(bull_evt)})",
                "source":          "task1_news",
                "direction":       "bullish",
                "baseline_weight": round(w, 3),
                "raw_value":       bull_evt,
                "evidence": [f"Detected: {', '.join(bull_evt)}",
                             f"{ec['bull_today']:.0f} event-tagged articles today vs "
                             f"{ec['bull_trail']:.1f}/day over the previous "
                             f"{ec['trail_days']} news days (+{bull_x:.1f}) [EV1]"]
            })
            bullish_score += w

        if bear_evt and bear_ok:
            w = _get_weight("n_bearish_events") * EVENT_SCALE   # [MW2]
            features.append({
                "factor":          f"Unusual bearish corporate event activity ({len(bear_evt)})",
                "source":          "task1_news",
                "direction":       "bearish",
                "baseline_weight": round(w, 3),
                "raw_value":       bear_evt,
                "evidence": [f"Detected: {', '.join(bear_evt)}",
                             f"{ec['bear_today']:.0f} event-tagged articles today vs "
                             f"{ec['bear_trail']:.1f}/day over the previous "
                             f"{ec['trail_days']} news days (+{bear_x:.1f}) [EV1]"]
            })
            bearish_score += w

    # ── Political backdrop tone: uses the political_direction feature ──
    # political_direction = mean FinBERT score of that day's POLITICAL news.
    # Weight comes from Ridge and is discounted by MW1 (it's market-wide).
    pol     = context.get("political", {})
    p_dir   = pol.get("avg_sentiment", 0.0)
    p_total = pol.get("total", 0)
    if "political_direction" in DATA_DRIVEN_WEIGHTS and p_total > 5 and abs(p_dir) > 0.10: # sentiment strong
        w_pol = _get_weight("political_direction")
        direction = "bullish" if p_dir > 0 else "bearish"
        features.append({
            "factor":          f"{'Positive' if p_dir > 0 else 'Negative'} geopolitical backdrop",
            "source":          "task1_political",
            "direction":       direction,
            "baseline_weight": round(w_pol, 3),
            "raw_value":       round(p_dir, 3),
            "evidence": [
                f"Avg political sentiment: {p_dir:+.2f} across {p_total} articles",
                "Weight: Ridge (political_direction), MW1-discounted"
            ]
        })
        if direction == "bullish":
            bullish_score += w_pol
        else:
            bearish_score += w_pol

    # ── Political news VOLUME [MW5] ─────────────────────────────────────────
    # Ridge's second-largest weight (0.93, bearish) had NO rule at all in the
    # previous version. It is the learned counterweight to political_direction:
    # heavy political coverage signals uncertainty regardless of tone. Its
    # sign HELD in Part A's temporal validation (−0.130 → −0.134), so unlike
    # the macro features this weight generalises. Restoring it re-creates the
    # offsetting pair Ridge actually learned — without it, positive political
    # tone pushed every stock bullish with nothing pushing back.
    #
    # Threshold = training-set mean of n_political (data-defined "above normal").
    _pol_mean = TASK5_ARTIFACTS.get("feature_means", {}).get("n_political")
    if _pol_mean and p_total > _pol_mean:
        w = _get_weight("n_political")
        features.append({
            "factor":          "Elevated political news volume",
            "source":          "task1_political",
            "direction":       "bearish",
            "baseline_weight": round(w, 3),
            "raw_value":       p_total,
            "evidence": [
                f"{p_total} political articles vs training average {_pol_mean:.0f}",
                "Ridge learned volume as a bearish signal (uncertainty proxy)"
            ]
        })
        bearish_score += w  #A surge in political news
        #= increased uncertainty = market preference for risk aversion = pressure on stocks

    # ── Technical signals (Task 2) ──
    tech = context.get("technical")
    if tech:
        # RSI: oversold (<30) bullish bounce; overbought (>70) bearish pullback
        if tech["rsi_signal"] == "oversold":
            w = _get_weight("rsi")
            features.append({
                "factor":          "RSI oversold — potential bounce",
                "source":          "task2_technical",
                "direction":       "bullish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["rsi"],
                "evidence": [f"RSI={tech['rsi']} (below 30 threshold, Wilder 1978)"]
            })
            bullish_score += w
        elif tech["rsi_signal"] == "overbought":
            w = _get_weight("rsi")
            features.append({
                "factor":          "RSI overbought — potential pullback",
                "source":          "task2_technical",
                "direction":       "bearish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["rsi"],
                "evidence": [f"RSI={tech['rsi']} (above 70 threshold, Wilder 1978)"]
            })
            bearish_score += w

        # MACD crossovers (Appel 1979)
        if tech["macd_signal"] == "bullish_crossover":
            w = _get_weight("macd_bullish_crossover")
            features.append({
                "factor":          "MACD bullish crossover",
                "source":          "task2_technical",
                "direction":       "bullish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["macd_value"],
                "evidence": [f"MACD line crossed above signal (value: {tech['macd_value']})"]
            })
            bullish_score += w
        elif tech["macd_signal"] == "bearish_crossover":
            w = _get_weight("macd_bearish_crossover")
            features.append({
                "factor":          "MACD bearish crossover",
                "source":          "task2_technical",
                "direction":       "bearish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["macd_value"],
                "evidence": [f"MACD line crossed below signal (value: {tech['macd_value']})"]
            })
            bearish_score += w

        # Bollinger Bands (Bollinger 2001)
        if tech["bb_position"] == "above_upper":
            w = _get_weight("bb_above_upper")
            features.append({
                "factor":          "Price above Bollinger upper band",
                "source":          "task2_technical",
                "direction":       "bullish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["bb_position"],
                "evidence": ["Momentum breakout signal (2σ above 20-day mean)"]
            })
            bullish_score += w
        elif tech["bb_position"] == "below_lower":
            w = _get_weight("bb_below_lower")
            features.append({
                "factor":          "Price below Bollinger lower band",
                "source":          "task2_technical",
                "direction":       "bearish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["bb_position"],
                "evidence": ["Momentum breakdown signal (2σ below 20-day mean)"]
            })
            bearish_score += w

        # VWAP intraday bias
        if tech["vwap_position"] == "above":
            w = _get_weight("vwap_above")
            features.append({
                "factor":          "Price above VWAP",
                "source":          "task2_technical",
                "direction":       "bullish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["vwap_position"],
                "evidence": ["Positive intraday bias"]
            })
            bullish_score += w

        # Weekly timeframe confirmation
        if tech["weekly_signal"] == "bullish":
            w = _get_weight("weekly_bullish")
            features.append({
                "factor":          "Weekly timeframe confirms bullish",
                "source":          "task2_technical",
                "direction":       "bullish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["weekly_signal"],
                "evidence": ["Multi-timeframe alignment (daily + weekly agree)"]
            })
            bullish_score += w
        elif tech["weekly_signal"] == "bearish":
            w = _get_weight("weekly_bullish")
            features.append({
                "factor":          "Weekly timeframe confirms bearish",
                "source":          "task2_technical",
                "direction":       "bearish",
                "baseline_weight": round(w, 3),
                "raw_value":       tech["weekly_signal"],
                "evidence": ["Multi-timeframe alignment (both bearish)"]
            })
            bearish_score += w

    # ── Macro regime (Task 3, market-wide) ──
    # Expert "gravity" multipliers documented as NON-data-driven: linear Ridge
    # cannot capture regime-specific severity, so these are explicit priors.
    macro = context.get("macro")
    if macro:
        regime = macro.get("regime", "Neutral")
        w      = _get_weight("regime_encoded")
        sector = context.get("sector", "")

        if regime == "Recessionary":
            features.append({
                "factor":          "Recessionary macro regime",
                "source":          "task3_macro",
                "direction":       "bearish",
                "baseline_weight": round(w * 2, 3),   # ⚠️ business gravity ×2.0
                "raw_value":       regime,
                "evidence": [
                    f"Yield spread inverted (spread={macro.get('yield_spread')}%)",
                    "Gravity ×2.0: historical recession precursor (NY Fed model)"
                ]
            })
            bearish_score += w * 2
        elif regime == "Risk-Off":
            features.append({
                "factor":          "Risk-Off macro regime (VIX elevated)",
                "source":          "task3_macro",
                "direction":       "bearish",
                "baseline_weight": round(w * 1.5, 3),   # ⚠️ business gravity ×1.5
                "raw_value":       regime,
                "evidence": [
                    f"VIX={macro.get('vix')} above 30 threshold",
                    "Gravity ×1.5: crisis-prone regime"
                ]
            })
            bearish_score += w * 1.5
        elif regime == "Risk-On":
            features.append({
                "factor":          "Risk-On macro regime",
                "source":          "task3_macro",
                "direction":       "bullish",
                "baseline_weight": round(w, 3),
                "raw_value":       regime,
                "evidence": [f"VIX={macro.get('vix')} below 15 (low fear)"]
            })
            bullish_score += w

        # High Rate regime — sector-dependent [B1]
        elif regime == "High Rate / Elevated Inflation":
            is_bank = _match_sector(sector, ["bank", "financial"])
            is_reit = _match_sector(sector, ["reit", "estate", "property", "trust"])
            if is_bank:
                features.append({
                    "factor":          "Bank sector benefits from high rates",
                    "source":          "task3_macro",
                    "direction":       "bullish",
                    "baseline_weight": round(w * 1.2, 3),   # ⚠️ sector gravity ×1.2
                    "raw_value":       regime,
                    "evidence": [
                        "Higher SORA expands net interest margins (bullish for banks)",
                        "Aharony & Swary (1980) — bank profitability responds positively to rate hikes"
                    ]
                })
                bullish_score += w * 1.2
            elif is_reit:
                features.append({
                    "factor":          "REIT sector pressured by high rates",
                    "source":          "task3_macro",
                    "direction":       "bearish",
                    "baseline_weight": round(w * 1.2, 3),
                    "raw_value":       regime,
                    "evidence": [
                        "Higher discount rates compress REIT valuations",
                        "Chan et al. (1990) — REIT prices inversely correlate with real rates"
                    ]
                })
                bearish_score += w * 1.2

        # Low Rate regime — inverse sector interaction [B1]
        elif regime == "Low Rate / Low Inflation":
            is_bank = _match_sector(sector, ["bank", "financial"])
            is_reit = _match_sector(sector, ["reit", "estate", "property", "trust"])
            if is_bank:
                features.append({
                    "factor":          "Low-rate regime pressures bank NIMs",
                    "source":          "task3_macro",
                    "direction":       "bearish",
                    "baseline_weight": round(w, 3),
                    "raw_value":       regime,
                    "evidence": [f"SORA: {macro.get('sora')}% (margin compression)"]
                })
                bearish_score += w
            elif is_reit:
                features.append({
                    "factor":          "Low-rate regime supports REIT valuations",
                    "source":          "task3_macro",
                    "direction":       "bullish",
                    "baseline_weight": round(w, 3),
                    "raw_value":       regime,
                    "evidence": [f"SORA: {macro.get('sora')}% (supportive of REIT distributions)"]
                })
                bullish_score += w

    # ── Options positioning (Task 4, market-wide) ──
    # P/C ratio (McMillan 2002): <0.7 calls dominate (bullish); >1.3 puts (bearish)
    opt = context.get("options")
    if opt:
        if opt["signal"] == "bullish":
            w = _get_weight("options_bullish")
            features.append({
                "factor":          "Options market bullish positioning",
                "source":          "task4_options",
                "direction":       "bullish",
                "baseline_weight": round(w, 3),
                "raw_value":       opt["pc_volume_ratio"],
                "evidence": [f"P/C Volume ratio={opt['pc_volume_ratio']} (calls dominate, <0.7)"]
            })
            bullish_score += w
        elif opt["signal"] == "bearish":
            w = _get_weight("options_bearish")
            features.append({
                "factor":          "Options market bearish positioning",
                "source":          "task4_options",
                "direction":       "bearish",
                "baseline_weight": round(w, 3),
                "raw_value":       opt["pc_volume_ratio"],
                "evidence": [f"P/C Volume ratio={opt['pc_volume_ratio']} (puts dominate, >1.3)"]
            })
            bearish_score += w

    # ── Aggregate sentiment + signal strength ──
    net_score    = bullish_score - bearish_score
    total_weight = bullish_score + bearish_score

    # Direction: sentiment threshold [MW3] is a tuned constant, not learned.
    if net_score > NET_SCORE_THRESHOLD:
        baseline_sentiment = "bullish"
    elif net_score < -NET_SCORE_THRESHOLD:
        baseline_sentiment = "bearish"
    else:
        baseline_sentiment = "neutral"

    # Signal strength: heuristic based on how one-sided the score is.
    # NOTE: this is SIGNAL CONCENTRATION, not a calibrated probability.
    # Calibrated probability belongs to Task 6 (Platt scaling / Isotonic).
    # The smoothing prior (+0.5 in denominator) prevents thin evidence from
    # producing 100% confidence.
    if baseline_sentiment == "neutral":
        baseline_confidence = BASE_CONFIDENCE
    else:
        dominant = max(bullish_score, bearish_score)
        boost    = (dominant / (total_weight + 0.5)) * (MAX_CONFIDENCE - BASE_CONFIDENCE)
        baseline_confidence = min(BASE_CONFIDENCE + boost, MAX_CONFIDENCE)

    features.sort(key=lambda f: f["baseline_weight"], reverse=True)

    # [C] Per-feature context so the LLM can talk about proportions honestly:
    #   share_of_direction = this feature's fraction of its own side's total
    #   flips_call         = removing it would push net_score across the
    #                        threshold (i.e. it is decisive, not just a caveat)
    for f in features:
        side_total = bullish_score if f["direction"] == "bullish" else bearish_score
        f["share_of_direction"] = round(f["baseline_weight"] / side_total, 2) if side_total > 0 else 0.0
        _net_without = net_score - f["baseline_weight"] if f["direction"] == "bullish" \
                        else net_score + f["baseline_weight"]
        f["flips_call"] = (
            (net_score >  NET_SCORE_THRESHOLD) != (_net_without >  NET_SCORE_THRESHOLD) or
            (net_score < -NET_SCORE_THRESHOLD) != (_net_without < -NET_SCORE_THRESHOLD)
        )

    return {
        "features":            features,
        "baseline_sentiment":  baseline_sentiment,
        "baseline_confidence": round(baseline_confidence, 3),
        "score_breakdown": {
            "bullish_score": round(bullish_score, 3),
            "bearish_score": round(bearish_score, 3),
            "net_score":     round(net_score, 3),
        }
    }

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: compute_shap_explanation  (Layer 2 — independent SHAP)
#
# Purpose: run the LogReg + RF models trained in Part A (on next-day returns,
# independent of the rules) and explain each prediction with SHAP.
#
# [#7] SHAP direction is derived from the model's PREDICTION, not raw sign.
# SHAP > 0 means "supports the predicted class". So for a bearish prediction,
# SHAP > 0 is a bearish signal — the naive sign(shap) logic was wrong for all
# bearish cases. Uses the SAME extract_features_row as Part A (no drift).
# ═══════════════════════════════════════════════════════════════════════════

def compute_shap_explanation(ticker, date, artifacts=None):
    """
    Compute SHAP values for LogReg + RF predictions on one (ticker, date).

    Returns per-feature SHAP contributions from both models plus a consensus
    list (features both models agree on), or None if no technical data.
    """
    if artifacts is None:
        artifacts = TASK5_ARTIFACTS

    # SAME extractor as Part A training (single source of truth, no drift).
    # return_baseline_sentiment=True gives us the rule-based sentiment too.
    features, _ = extract_features_row(ticker, date, return_baseline_sentiment=True)
    if features is None:
        return None

    # Build feature vector in EXACT training order (order matters for models)
    feature_names = artifacts["feature_names"]
    X_row = np.array([[features[fn] for fn in feature_names]])

    # Apply the SAME scaler fitted during training (LogReg needs scaled input)
    X_row_scaled = artifacts["scaler"].transform(X_row)

    # ── Predict with both models [DD5: keys are *_return, matching Part A] ──
    logreg    = artifacts["logreg_return"]
    rf        = artifacts["rf_return"]
    label_enc = artifacts["label_encoder"]

    lr_pred_idx = logreg.predict(X_row_scaled)[0]    # 0/1/2
    rf_pred_idx = rf.predict(X_row)[0]

    lr_pred = label_enc.inverse_transform([lr_pred_idx])[0]   # "bullish"/"neutral"/"bearish"
    rf_pred = label_enc.inverse_transform([rf_pred_idx])[0]

    # predict_proba of the predicted class → self-reported confidence
    lr_probs = logreg.predict_proba(X_row_scaled)[0]
    rf_probs = rf.predict_proba(X_row)[0]
    lr_conf  = float(lr_probs[lr_pred_idx])
    rf_conf  = float(rf_probs[rf_pred_idx])

    # ── SHAP values (version-tolerant shape handling) ──
    lr_shap_vals = artifacts["logreg_explainer"].shap_values(X_row_scaled)
    rf_shap_vals = artifacts["rf_explainer"].shap_values(X_row)

    def _extract_shap_for_class(shap_vals, pred_idx):
        """Return 1D SHAP array for the predicted class across shap versions."""
        if isinstance(shap_vals, list):
            return np.array(shap_vals[pred_idx][0])       # legacy list-of-2D
        elif shap_vals.ndim == 3:
            return shap_vals[0, :, pred_idx]               # new 3D (shap >= 0.42)
        else:
            return shap_vals[0]                            # binary / single-output

    lr_shap = _extract_shap_for_class(lr_shap_vals, lr_pred_idx)
    rf_shap = _extract_shap_for_class(rf_shap_vals, rf_pred_idx)

    # ── Per-feature summary with correct direction [#7] ──
    feature_shap = []
    for i, fname in enumerate(feature_names):
        lr_val  = float(lr_shap[i])
        rf_val  = float(rf_shap[i])
        avg_val = (lr_val + rf_val) / 2

        # Direction = combine SHAP sign WITH the model's prediction.
        # SHAP > 0 supports the predicted class; translate to bullish/bearish.
        if lr_pred == "bullish":
            direction = "bullish" if avg_val > 0 else "bearish"
        elif lr_pred == "bearish":
            direction = "bearish" if avg_val > 0 else "bullish"
        else:  # neutral prediction — SHAP sign heuristic
            direction = "bullish" if avg_val > 0 else "bearish"

        feature_shap.append({
            "feature":         fname,
            "label":           _map_to_readable(fname),
            "logreg_shap":     round(lr_val, 4),
            "rf_shap":         round(rf_val, 4),
            "avg_shap":        round(avg_val, 4),
            "direction":       direction,
            "predicted_class": lr_pred,
        })

    # ── Consensus: both models same-sign SHAP AND |avg| > 0.01 (skip noise) ──
    consensus = [
        f for f in feature_shap
        if (f["logreg_shap"] > 0) == (f["rf_shap"] > 0) and abs(f["avg_shap"]) > 0.01
    ]
    consensus.sort(key=lambda x: abs(x["avg_shap"]), reverse=True)

    return {
        "predicted_sentiment_logreg":  lr_pred,
        "predicted_confidence_logreg": round(lr_conf, 3),
        "predicted_sentiment_rf":      rf_pred,
        "predicted_confidence_rf":     round(rf_conf, 3),
        "all_features_shap":           feature_shap,
        "top_features_consensus":      consensus[:5],
    }

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: detect_contradictions  (Layer 3 — uncertainty)
#
# Three disagreement signals, each SYMMETRIC (like compared with like):
#
#   internal      A rule fires against the rule layer's own baseline
#                 (e.g. bullish outlook but RSI overbought triggered).
#                 Expected in any multi-factor score; useful as a caveat.
#   cross-model   Both ML models AGREE on a call that differs from the
#                 rule baseline. Aggregate vs aggregate — the old version
#                 compared a single SHAP feature against the whole rule
#                 score, which was apples-to-oranges.
#   model-split   The two ML models disagree WITH EACH OTHER. Linear and
#                 non-linear views of the same data diverge. Was computed
#                 previously and silently discarded.
#
# Also reports consensus_ratio — the fraction of SHAP features on which both
# models agree in sign. A low ratio means the second opinion itself is shaky.
# ═══════════════════════════════════════════════════════════════════════════

def detect_contradictions(features_result, shap_result=None):
    """
    Surface disagreement across the pipeline. shap_result may be None →
    only internal tension is detected.
    """
    contradictions = []
    baseline = features_result["baseline_sentiment"]

    # internal: rule layer's own tension
    if baseline in ("bullish", "bearish"):
        opposite = "bearish" if baseline == "bullish" else "bullish"
        for f in features_result["features"]:
            if f["direction"] == opposite:
                contradictions.append({
                    "type":   "internal",
                    "factor": f["factor"],
                    "weight": f["baseline_weight"],
                    "note":   f"{opposite.capitalize()} feature within {baseline} outlook",
                })

    n_cross = n_split = 0
    consensus_ratio = None

    if shap_result is not None:
        lr = shap_result["predicted_sentiment_logreg"]
        rf = shap_result["predicted_sentiment_rf"]

        # cross-model: both ML models agree, and disagree with the rule layer
        if lr == rf and lr != baseline:
            avg_conf = (shap_result["predicted_confidence_logreg"] +
                        shap_result["predicted_confidence_rf"]) / 2
            contradictions.append({
                "type":      "cross-model",
                "factor":    "ML second opinion vs rule baseline",
                "weight":    round(avg_conf, 3),
                "rule_says": baseline,
                "ml_says":   lr,
                "note":      f"Both ML models read {lr}; rules read {baseline}",
            })
            n_cross = 1

        # model-split: the two ML models disagree with each other
        elif lr != rf:
            contradictions.append({
                "type":   "model-split",
                "factor": "LogReg vs RandomForest",
                "weight": round(abs(shap_result["predicted_confidence_logreg"] -
                                    shap_result["predicted_confidence_rf"]), 3),
                "note":   f"Linear model reads {lr}, non-linear reads {rf}",
            })
            n_split = 1

        # consensus_ratio: how much of the second opinion is solid?
        all_shap = shap_result.get("all_features_shap", [])
        if all_shap:
            n_agree = sum(1 for f in all_shap
                          if (f["logreg_shap"] > 0) == (f["rf_shap"] > 0))
            consensus_ratio = round(n_agree / len(all_shap), 2)

    n_internal = sum(1 for c in contradictions if c["type"] == "internal")

    # Warning level: cross-model heaviest, split medium, internal light.
    if n_cross and n_internal >= 2:
        warning_level = "high"
    elif n_cross or n_split or n_internal >= 2:
        warning_level = "mild"
    else:
        warning_level = "low"

    # Shaky second opinion (models agree on <40% of feature signs) → mild
    if consensus_ratio is not None and consensus_ratio < 0.4 and warning_level == "low":
        warning_level = "mild"

    return {
        "contradictions":    contradictions,
        "count":             len(contradictions),
        "internal_count":    n_internal,
        "cross_model_count": n_cross,
        "model_split_count": n_split,
        "consensus_ratio":   consensus_ratio,
        "warning_level":     warning_level,
    }

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: build_llm_prompt  (Layer 4 — prompt assembly)
#
# Purpose: assemble a prompt that gives the LLM enough grounded information
# (Layer 1 features, actual news headlines + content, Layer 2 SHAP, Layer 3
# contradictions, macro context) while strictly constraining it: it may not
# reverse the sentiment, may only nudge confidence ±0.20, may not invent facts.
#
# [B3] sector_impact is passed in full (Task 3 already writes concise text).
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# STEP 11: call_groq_llm  (Layer 4 — inference)
#
# Purpose: call Groq (Llama 3.1 8B), parse JSON, absorb failures.
# [B4] 429 backoff base = 3s → retry timing 3, 6, 12, 24, 48s (free-tier safe).
# Never throws — returns None on any failure so the caller can fall back.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# STEP 12: validate_llm_output  (Layer 4 — guardrail)
#
# Purpose: treat the LLM as an untrusted input. Reject output that misses keys,
# reverses the sentiment, or moves confidence out of ±0.20. On failure the
# caller falls back to a rule-based summary.
# [B5] Sentiment check is case-insensitive with auto-normalization.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# STEP 13: store_llm_analysis
#
# Purpose: upsert the final result to llm_analyses (one row per
# ticker+analysis_date) and append an immutable audit snapshot to
# llm_analyses_audit (full pipeline data + prompt hash for traceability).
#
# ⚠️ Schema note: llm_analyses uses analysis_date / model_used; the audit table
# uses input_hash / full_input / full_output (JSONB) / validation_passed.
# ═══════════════════════════════════════════════════════════════════════════

def store_llm_analysis(ticker, date, llm_output, features_result, contradictions,
                       shap_result, prompt, is_valid, notes):
    """Store final result (UPSERT) + immutable audit log (APPEND-ONLY)."""
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    _meta        = llm_output.get("_meta", {})
    model_used   = _meta.get("model", "unknown")
    total_tokens = _meta.get("tokens", 0)
    input_tokens  = int(total_tokens * 0.7) if total_tokens else 0
    output_tokens = total_tokens - input_tokens

    # ── llm_analyses (final consumable result) ──
    row = {
        "ticker":              ticker,
        "analysis_date":       date,
        "sentiment":           llm_output.get("sentiment", features_result["baseline_sentiment"]),
        "confidence":          float(llm_output.get("confidence", features_result["baseline_confidence"])),
        "sentiment_score":     float(llm_output.get("sentiment_score", features_result["score_breakdown"]["net_score"])),
        "key_drivers":         json.dumps(llm_output.get("key_drivers", [])),
        "risk_factors":        json.dumps(llm_output.get("risk_factors", [])),
        "summary":             llm_output.get("summary", ""),
        "baseline_sentiment":  features_result["baseline_sentiment"],
        "baseline_confidence": features_result["baseline_confidence"],
        "warning_level":       contradictions["warning_level"],
        "n_contradictions":    contradictions["count"],
        "is_valid":            is_valid,
        "validation_notes":    "; ".join(notes),
        "model_used":          model_used,
        "prompt_hash":         prompt_hash,
        "source_count":        len(features_result.get("features", [])),
        "data_freshness":      date,
    }

    try:
        with SUPABASE_ENGINE.begin() as conn:
            conn.execute(text("""
                INSERT INTO llm_analyses (
                    ticker, analysis_date, sentiment, confidence, sentiment_score,
                    key_drivers, risk_factors, summary,
                    baseline_sentiment, baseline_confidence,
                    warning_level, n_contradictions,
                    is_valid, validation_notes, model_used, prompt_hash,
                    source_count, data_freshness
                )
                VALUES (
                    :ticker, :analysis_date, :sentiment, :confidence, :sentiment_score,
                    CAST(:key_drivers AS JSONB), CAST(:risk_factors AS JSONB), :summary,
                    :baseline_sentiment, :baseline_confidence,
                    :warning_level, :n_contradictions,
                    :is_valid, :validation_notes, :model_used, :prompt_hash,
                    :source_count, :data_freshness
                )
                ON CONFLICT (ticker, analysis_date) DO UPDATE SET
                    sentiment           = EXCLUDED.sentiment,
                    confidence          = EXCLUDED.confidence,
                    sentiment_score     = EXCLUDED.sentiment_score,
                    key_drivers         = EXCLUDED.key_drivers,
                    risk_factors        = EXCLUDED.risk_factors,
                    summary             = EXCLUDED.summary,
                    baseline_sentiment  = EXCLUDED.baseline_sentiment,
                    baseline_confidence = EXCLUDED.baseline_confidence,
                    warning_level       = EXCLUDED.warning_level,
                    n_contradictions    = EXCLUDED.n_contradictions,
                    is_valid            = EXCLUDED.is_valid,
                    validation_notes    = EXCLUDED.validation_notes,
                    model_used          = EXCLUDED.model_used,
                    prompt_hash         = EXCLUDED.prompt_hash
            """), row)
    except Exception as e:
        print(f"    ⚠️  llm_analyses upsert failed: {type(e).__name__}: {e}")

    # ── llm_analyses_audit (immutable full snapshot) ──
    audit_row = {
        "ticker":              ticker,
        "analysis_date":       date,
        "input_hash":          prompt_hash,
        "model_used":          model_used,
        "input_tokens":        input_tokens,
        "output_tokens":       output_tokens,
        "baseline_sentiment":  features_result["baseline_sentiment"],
        "final_sentiment":     llm_output.get("sentiment", features_result["baseline_sentiment"]),
        "baseline_confidence": features_result["baseline_confidence"],
        "final_confidence":    float(llm_output.get("confidence", features_result["baseline_confidence"])),
        "full_input":          json.dumps({"prompt": prompt}),
        "full_output":         json.dumps(llm_output),
        "validation_passed":   is_valid,
        "validation_notes":    "; ".join(notes),
        "features_result":     json.dumps({
            **features_result,
            # [v7] Rule-layer provenance. The rule layer changes independently
            # of the Part A model (MW1 gained regime_encoded on 3 Sep without
            # any retrain), so model_version alone cannot tell you which rules
            # produced a row. Stamped into the existing JSONB rather than added
            # as a column, so no schema migration is needed.
            "_rule_layer_version":  RULE_LAYER_VERSION,
            "_model_version":       TASK5_ARTIFACTS.get("model_version"),   # [22 Sep]
            "_market_wide_features": sorted(MARKET_WIDE_FEATURES),
            "_market_wide_scale":    MARKET_WIDE_SCALE,
            "_net_score_threshold":  NET_SCORE_THRESHOLD,
        }),
        "contradictions":      json.dumps(contradictions),
        "shap_result":         json.dumps({
            "predicted_sentiment_logreg":  shap_result["predicted_sentiment_logreg"],
            "predicted_confidence_logreg": shap_result["predicted_confidence_logreg"],
            "predicted_sentiment_rf":      shap_result["predicted_sentiment_rf"],
            "predicted_confidence_rf":      shap_result["predicted_confidence_rf"],
            "top_features_consensus":       shap_result["top_features_consensus"],
        }),
    }

    try:
        with SUPABASE_ENGINE.begin() as conn:
            conn.execute(text("""
                INSERT INTO llm_analyses_audit (
                    ticker, analysis_date, input_hash, model_used,
                    input_tokens, output_tokens,
                    baseline_sentiment, final_sentiment,
                    baseline_confidence, final_confidence,
                    full_input, full_output,
                    validation_passed, validation_notes,
                    features_result, contradictions, shap_result
                )
                VALUES (
                    :ticker, :analysis_date, :input_hash, :model_used,
                    :input_tokens, :output_tokens,
                    :baseline_sentiment, :final_sentiment,
                    :baseline_confidence, :final_confidence,
                    CAST(:full_input AS JSONB), CAST(:full_output AS JSONB),
                    :validation_passed, :validation_notes,
                    CAST(:features_result AS JSONB), CAST(:contradictions AS JSONB), CAST(:shap_result AS JSONB)
                )
            """), audit_row)
    except Exception as e:
        print(f"    ⚠️  audit log failed: {type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
print("✅ Part B loaded")
print(f"   model_version       = {TASK5_ARTIFACTS.get('model_version', '?')}  [T5_MODEL_VERSION]  "
      f"(trained {TASK5_ARTIFACTS.get('train_window', '?')})")
print(f"   net_score_threshold = ±{NET_SCORE_THRESHOLD:.2f}  [MW3, tuned]")
print(f"   base_confidence     = {BASE_CONFIDENCE:.3f}   (from Part A)")
print(f"   market_wide_scale   = {MARKET_WIDE_SCALE}    [MW1, temporal-validation informed]")
print(f"   event_scale         = {EVENT_SCALE}    [MW2, event direction unreliable]")
print(f"   new rules           = n_political (bearish counterweight) [MW5]")
print(f"   rule_layer_version  = {RULE_LAYER_VERSION}   [EV1: event rules fire only when "
      f"today's event articles ≥ trailing {EVENT_TRAIL_DAYS}-day mean + {EVENT_EXCESS_MIN:.0f}]")
print(f"   Layer 3             = internal + cross-model + model-split + consensus_ratio")