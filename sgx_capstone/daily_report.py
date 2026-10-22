# ═══════════════════════════════════════════════════════════════════════════
# TASK 7  —  DAILY SIGNAL REPORT  (v8, 19 Oct 2026)
#
# The investor-facing report, on the unchanged Task 7 engine (report input,
# 11 patterns, attention, withholding rule, compliance checks R1-R7). Four
# pages for an investor, each answering one question, and one for reviewers:
#
#   1  Today                  today in 30 seconds, headline numbers, which
#                             parts can be relied on, all 15 stocks ranked by
#                             today's news and signals (news and chart apart)
#   2  Last 5 trading days    a stocks-by-days chart by sector; cards for the
#                             biggest movers: what happened, the Task 5 AI
#                             summary (live), today's news and headlines, price
#                             chart, the Task 5 reading and every rule it used,
#                             whether the news explains the move
#   3  Next 3 trading days    Task 6 forecast for every stock with its record
#                             against a simple rule, normal swing (also per
#                             S$1,000), calmer than most; what these are worth
#   4  System                 the seven tasks behind the report: Task 5's four
#                             layers, Task 6's weekly champion-challenger, and
#                             today's 11-pattern breakdown
#   5  Findings               what three months of testing show (one table,
#                             all stocks), limitations, glossary, disclosure
#   5  For reviewers          how the report was produced, provenance, exact
#                             attributions, rules
#
# Every number and effect direction is written by template. The language
# model's only text is the Task 5 news summary (live report), checked by Task
# 5's guardrails and re-checked here.
#
#   python -m sgx_capstone sample [--date 2026-09-18]        offline, 17 Sep 2026
#   python -m sgx_capstone signal-report --date YYYY-MM-DD   live database
#   ... signal-report --date YYYY-MM-DD --save-snapshot     also freeze the inputs so that
#   python -m sgx_capstone sample --date YYYY-MM-DD          rebuilds it offline
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import pickle
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from . import outlook as O                                        # noqa: E402
from .research.r14_outlook import SESSION                        # noqa: E402

SGT = timezone(timedelta(hours=8))
# Default sample: the saved copy of the live database for 21 Sep (the first day
# with both the rl_v8 rule fix and out-of-fold calibrated forecasts live).
SAMPLE_DATE = "2026-09-21"
# The stored-records sample (no database copy, Task 5 re-run from feature rows):
# 17 Sep, not 18 Sep, because on 18 Sep the news collection returned 1 SGX Group
# article (50 on 17 Sep), too few to read.
RECORDS_DATE = "2026-09-17"
SAMPLE_DATE_NOTE = (" Stored-records date 17 Sep: on 18 Sep the news collection returned 1 SGX Group article "
                    "(50 on 17 Sep), too few to read.")
PERIOD_SESSIONS = 5
BLUE, ORANGE, RED, GREY, LIGHT = "#2a78d6", "#eb6834", "#e34948", "#898781", "#c3c2b7"
INK2, GRID = "#52514e", "#e1e0d9"
NAVY = (0x1a, 0x2e, 0x55)

EVIDENCE_WORD = {"bullish": "Positive", "bearish": "Negative", "neutral": "Neutral"}
FORECAST_WORD = {"bullish": "Rise", "bearish": "Fall", "neutral": "No big move"}
PATTERN_NAME = {
    "momentum": "Momentum", "mean_reversion": "Mean reversion", "move_complete": "Move complete",
    "muted_pending": "Muted, pending", "muted_fading": "Muted, fading", "dormant": "Dormant",
    "lag": "Lag", "contradicted": "Contradicted", "noise": "Noise", "quiet": "Quiet",
    "model_only": "Model-only view", "price_unavailable": "Price unavailable",
}
PATTERN_WORD = {
    "momentum": "news and price agreed; the model expects the move to continue",
    "mean_reversion": "news and price agreed, but the model expects a pullback",
    "move_complete": "the signals played out today; nothing further expected",
    "muted_pending": "signals leaned one way, price barely moved; the model still expects that direction",
    "muted_fading": "signals leaned one way, price barely moved; the model expects the opposite",
    "dormant": "a slight lean in the signals; neither price nor model followed",
    "lag": "price went against the signals; the model sides with the signals",
    "contradicted": "price went against the signals and the model agrees with the price",
    "noise": "price went against the signals; the model has no clear view",
    "quiet": "no directional signal today",
    "model_only": "signals flat, but the model has a view",
    "price_unavailable": "no same-day price available",
}

# Task 5 rule names in plain words (rule layer rl_v8). Counts in brackets are
# dropped. Market-wide inputs are marked, because they are the same for all 15.
SIGNAL_WORDS = {
    "Positive company news sentiment": "more positive than negative company articles",
    "Negative company news sentiment": "more negative than positive company articles",
    "Unusual bullish corporate event activity": "more earnings, dividend or upgrade news than usual",
    "Unusual bearish corporate event activity": "more warning, downgrade or layoff news than usual",
    "Bullish corporate events": "earnings, dividend or upgrade headlines",
    "Bearish corporate events": "warning, downgrade or layoff headlines",
    "RSI oversold — potential bounce": "RSI below 30 (the price fell fast, 'oversold')",
    "RSI overbought — potential pullback": "RSI above 70 (the price rose fast, 'overbought')",
    "MACD bullish crossover": "short-term trend turned up (MACD)",
    "MACD bearish crossover": "short-term trend turned down (MACD)",
    "Price above Bollinger upper band": "price above its usual range (Bollinger band)",
    "Price below Bollinger lower band": "price below its usual range (Bollinger band)",
    "Price above VWAP": "price above its volume-weighted average",
    "Weekly timeframe confirms bullish": "weekly trend up",
    "Weekly timeframe confirms bearish": "weekly trend down",
    "Recessionary macro regime": "US yield curve signals recession risk (market-wide)",
    "Risk-Off macro regime (VIX elevated)": "high market fear, VIX above 30 (market-wide)",
    "Risk-On macro regime": "calm market, VIX below 15 (market-wide)",
    "Options market bullish positioning": "Singapore ETF options lean positive (market-wide)",
    "Options market bearish positioning": "Singapore ETF options lean negative (market-wide)",
}
# Not listed on stock cards (Appendix B): the political-tone rule reads the
# mean FinBERT confidence of political articles, which is always positive, so
# it fires every day whatever the news says; the volume rule is its fitted
# counterweight. Both are the same for every stock and largely cancel.
POLITICAL_RULES = ("geopolitical backdrop", "Elevated political news volume")
# Short labels for the stock board; * = market-wide (the same for all 15).
SIGNAL_SHORT = {
    "Positive company news sentiment": "Positive company news",
    "Negative company news sentiment": "Negative company news",
    "Unusual bullish corporate event activity": "More event news",
    "Unusual bearish corporate event activity": "More warning news",
    "Bullish corporate events": "Earnings/dividend headlines",
    "Bearish corporate events": "Warning/downgrade headlines",
    "RSI oversold — potential bounce": "RSI oversold",
    "RSI overbought — potential pullback": "RSI overbought",
    "MACD bullish crossover": "Trend turned up",
    "MACD bearish crossover": "Trend turned down",
    "Price above Bollinger upper band": "Above usual range",
    "Price below Bollinger lower band": "Below usual range",
    "Price above VWAP": "Above average price",
    "Weekly timeframe confirms bullish": "Weekly trend up",
    "Weekly timeframe confirms bearish": "Weekly trend down",
    "Recessionary macro regime": "Recession signal*",
    "Risk-Off macro regime (VIX elevated)": "High market fear*",
    "Risk-On macro regime": "Calm market*",
    "Options market bullish positioning": "ETF options positive*",
    "Options market bearish positioning": "ETF options negative*",
}
NEWS_SOURCES, CHART_SOURCES = ("task1_news",), ("task2_technical",)
ATTENTION_FLAG = {"ELEVATED": "⚑"}
GLOSSARY = [
    ["News & signals", "What today's news and indicators say on top of the price itself: positive, neutral or "
     "negative. It describes the day; the forecast is Task 6's job."],
    ["News today", "The company articles collected on the day, each read as positive, negative or neutral by a "
     "finance language model (FinBERT). 'Mostly positive' means clearly more positive than negative articles."],
    ["Normal day / big move", "A normal day: a one-day move within about ±0.6%, treated as the price not reacting. "
     "A big move (Task 6): a three-day change beyond about ±1%. Both are half a standard deviation of past moves."],
    ["RSI", "A 0-100 gauge of how fast the price has moved lately. Below 30 'oversold' (fell fast), above 70 "
     "'overbought' (rose fast)."],
    ["Average price (VWAP)", "The day's average price, weighted by how many shares traded at each price."],
    ["Usual range (Bollinger band)", "The 20-day average price plus or minus two standard deviations; the price "
     "stays inside it on most days, so a close outside it is unusual."],
    ["Trend turned (MACD)", "A short-term price average crossed a longer one."],
    ["Singapore ETF options", "Bets traders place on a fund that tracks Singapore shares; more 'puts' means more "
     "people buying protection against a fall. The same for all 15 stocks."],
    ["Normal 3-day swing", "How far the stock usually moves over three trading days (its recent daily volatility "
     "× √3); about 2 times in 3 the move stays inside it. A yardstick for whether a move is unusual."],
    ["Calmer than most", "The five stocks with the highest chance of no big move in Task 6's forecast for the day."],
    ["Task 6 lean", "The model's most likely outcome over the next three trading days: rise, fall or no big move, "
     "with the probability it gave. Probabilities are calibrated out-of-fold, so 50% means about half come true."],
    ["★  ⚑  ✓  ✗", "★ more company-event news than usual. ⚑ worth a closer look: the price went against the news, or "
     "the news is silent while the model holds a firm view, and the model has a usable record on this stock. ✓ / ✗ "
     "the model's record on that stock is about as good as or better than / worse than always guessing its most "
     "common past outcome."],
]
PERIOD_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten"}
WARNING_WORDS = {"low": "the system's internal checks found the signals consistent",  # only 'high' is printed
                 "mild": "the system's internal checks found the signals somewhat mixed",
                 "high": "the system's internal checks found the signals strongly mixed"}


# ═══════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def pct(x, dp=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{dp}f}%"


def short(name):
    for a, b in (("Commercial Trust", "Comm. Trust"), ("Industrial Trust", "Industrial"),
                 ("International", "Intl"), ("Integrated", "Integ.")):
        name = name.replace(a, b)
    return name


def evidence_vs_price(sentiment, move_class):
    """Axis 1 of the pattern engine in words, with the neutral case split."""
    if move_class is None:
        return "No price"
    if sentiment == "neutral":
        return "Moved without a signal" if move_class != "neutral" else "No signal, little move"
    if move_class == sentiment:
        return "Price agreed"
    if move_class == "neutral":
        return "Not yet in the price"
    return "Price went the other way"


def corporate_nouns(low):
    """'buy-out' / 'buyback' name a corporate event, not advice to buy; hide them from the advice check."""
    return re.sub(r"\bbuy[\s\-\u2010\u2011\u2013]?(?:outs?|backs?)\b", "corporate deal", low)


def summary_is_clean(text):
    """Second pass on the Task 5 summary: advice, target and certainty rules."""
    import task7_generate as g
    body = corporate_nouns((text or "").lower())
    return bool(body) and not any(
        re.search(p, body) for p in g._R1_PATTERNS + g._R2_PATTERNS + g._R3_PATTERNS)


_CONNECTOR = re.compile(r"^(however|nevertheless|moreover|still|but|also|additionally|furthermore),?\s+",
                        re.IGNORECASE)


def clean_t5_summary(text):
    """Task 5 summary as printed: in full, except sentences that read as advice
    ('investors should monitor ...') or quote Task 5's internal signal-strength
    number ('confidence of 0.55', easily misread as a probability); 'bullish' /
    'bearish' shown as 'positive' / 'negative'. Political sentences are kept: the
    political rule is flagged once in section 2 and in the limitations."""
    if not text:
        return None
    keep = [x for x in re.split(r"(?<=[.!?])\s+", text.strip())
            if x and not re.search(r"\b(should|recommend\w*|monitor|watch for)\b|confidence (?:of |at |level )?[\d.]",
                                   x, re.IGNORECASE)]
    if not keep:
        return None
    first = _CONNECTOR.sub("", keep[0])
    keep[0] = first[:1].upper() + first[1:]
    out = " ".join(keep)
    for a, b in (("SHAP values", "the ML models' explanations"), ("SHAP", "ML-model"),
                 ("bullishness", "a positive lean"), ("bearishness", "a negative lean"),
                 ("bullish", "positive"), ("Bullish", "Positive"), ("bearish", "negative"), ("Bearish", "Negative")):
        out = out.replace(a, b)
    return out


def strip_political_sentences(text):
    """Drop Task 5 summary sentences about political news.

    The political-tone rule reads FinBERT confidence, not tone (Appendix B),
    so a sentence such as "a supportive geopolitical backdrop" can contradict
    the political headlines of the day. Whole sentences are removed; a leading
    connector left dangling ("However, ...") is dropped. None if nothing is left.
    """
    if not text:
        return None
    keep = [s for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if s and not re.search(r"politic", s, re.IGNORECASE)]
    if not keep:
        return None
    first = _CONNECTOR.sub("", keep[0])
    keep[0] = first[:1].upper() + first[1:]
    return " ".join(keep)


def _base_factor(factor):
    return re.sub(r"\s*\(\d+\)\s*$", "", str(factor or "")).strip()


def plain_signal(factor):
    base = _base_factor(factor)
    return SIGNAL_WORDS.get(base, base[:1].lower() + base[1:])


def short_signal(factor):
    base = _base_factor(factor)
    return SIGNAL_SHORT.get(base, base)


def card_signals(signals):
    """Fired Task 5 rules without the two political-news rules (Limitations)."""
    return [s for s in signals or []
            if not any(p.lower() in str(s.get("factor", "")).lower() for p in POLITICAL_RULES)]


MARKET_SOURCES = ("task3_macro", "task4_options")


def market_signals(D):
    """Market-wide rules that fired today (identical for all 15 stocks)."""
    for r in D["rows"].values():
        m = [plain_signal(s["factor"]).replace(" (market-wide)", "") for s in card_signals(r.get("signals"))
             if s.get("source") in MARKET_SOURCES]
        if m:
            return m
    return []


def main_signal(signals, direction):
    """Heaviest stock-specific rule pointing one way (no political or
    market-wide rules), as a short label."""
    side = [s for s in card_signals(signals)
            if s.get("direction") == direction and s.get("source") not in MARKET_SOURCES]
    if not side:
        return "—"
    best = max(side, key=lambda s: float(s.get("weight") or 0.0))
    return short_signal(best["factor"])


def channel_agreement(signals):
    """Cross-channel consistency: do the news rules and the price-chart rules point the same way?"""
    def lean(sources):
        sc = sum(1 if s.get("direction") == "bullish" else -1
                 for s in card_signals(signals) if s.get("source") in sources)
        return (sc > 0) - (sc < 0)
    news, chart = lean(NEWS_SOURCES), lean(CHART_SOURCES)
    if news and chart:
        return ("Yes (up)" if news > 0 else "Yes (down)") if news == chart else "No (mixed)"
    if news:
        return "News only"
    if chart:
        return "Chart only"
    return "None"


def signals_text(signals):
    """'Pointing up: ...; pointing down: ...' from the fired Task 5 rules."""
    shown = card_signals(signals)
    up = [plain_signal(s["factor"]) for s in shown if s.get("direction") == "bullish"]
    down = [plain_signal(s["factor"]) for s in shown if s.get("direction") == "bearish"]
    if not up and not down:
        return None
    parts = []
    if up:
        parts.append("pointing up: " + "; ".join(up))
    if down:
        parts.append("pointing down: " + "; ".join(down))
    s = "Signals " + ". Signals ".join(parts) + "."
    return s


def _span(D):
    """'week' for the default five trading days, otherwise 'period'."""
    return "week" if D["period"]["sessions"] == 5 else "period"


def period_phrase(D):
    """'the five sessions 14-18 Sep' from the period stored in D."""
    first, end = pd.Timestamp(D["period"]["first"]), pd.Timestamp(D["period"]["end"])
    n = D["period"]["sessions"]
    span = (f"{first:%d}–{end:%d %b}" if first.month == end.month
            else f"{first:%d %b}–{end:%d %b}")
    return f"the {PERIOD_WORDS.get(n, str(n))} trading days {span}"


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return [v]
    if isinstance(v, dict):
        v = list(v.values())
    out = []
    for item in v:
        if isinstance(item, dict):
            item = item.get("factor") or item.get("description") or item.get("text") \
                or next(iter(item.values()), "")
        out.append(str(item))
    return [x for x in out if x.strip()]


def names_and_sectors(root):
    tick = json.loads((Path(root) / "settings" / "sgx.json").read_text())["tickers"]
    return ({t: v["name"] for t, v in tick.items()}, {t: v["sector"] for t, v in tick.items()})


def stored_daily_returns(root):
    """Close-to-close returns (percent) by session from the stored walk-forward
    file. The one-session return stored on date t belongs to the next session.
    A session without one is filled from a stored three- or five-session
    return when every other session in its window is known; the stored
    returns compound exactly (largest difference about 1e-14), so the fill is
    exact. Returns (frame indexed by session, number of sessions filled)."""
    from functools import reduce
    wf = pd.read_csv(Path(root) / "outputs" / "research" / "walk_forward_predictions.csv")
    wf = wf[(wf.calib == "oof") & (wf.model == "RandomF")]

    def nxt(d, k):
        return (pd.Timestamp(d) + k * SESSION).strftime("%Y-%m-%d")

    r = {nxt(d, 1): g.set_index("ticker").ret for d, g in wf[wf.h == 1].groupby("date")}
    windows = [(h, d, g.set_index("ticker").ret) for h in (3, 5) for d, g in wf[wf.h == h].groupby("date")]
    filled, changed = 0, True
    while changed:
        changed = False
        for h, d, ret in windows:
            sess = [nxt(d, k) for k in range(1, h + 1)]
            miss = [s for s in sess if s not in r]
            if len(miss) == 1:
                known = reduce(lambda a, b: a * b, [1 + r[s] / 100 for s in sess if s in r])
                r[miss[0]] = ((1 + ret / 100) / known - 1) * 100
                filled += 1
                changed = True
    return pd.DataFrame(r).T.sort_index(), filled


# ═══════════════════════════════════════════════════════════════════════════
# SHARED ASSEMBLY  (the same for the sample and the live report)
# ═══════════════════════════════════════════════════════════════════════════

def assemble(D, art, model, features, returns, ev):
    """Patterns, attention and outlook for every stock, from the inputs in D."""
    import task7_report as R
    probs = O.predict(art, model, features)
    looks = O.build_outlook(art, model, features, returns, ev, names=D["names"])
    t5_thr = R.T5_THRESHOLD_DEFAULT
    for t, r in D["rows"].items():
        cls = probs.loc[t].idxmax() if r.get("t6_class") is None else r["t6_class"]
        p = float(probs.loc[t].max()) if r.get("t6_prob") is None else r["t6_prob"]
        same = R.classify_actual(r["move"], t5_thr) if r["move"] is not None else None
        archetype, _, _ = R.classify_archetype(r["sentiment"], same, cls, p)
        edge = r.get("edge")
        track = {"t6_edge": edge, "t6_n": r.get("n_prior", 0),
                 "t6_accuracy": None if edge is None else 0.5 + edge, "t6_baseline": 0.5}
        level, _ = R.compute_attention(archetype, p, track, r.get("warning"))
        r.update({"t6_class": cls, "t6_prob": p, "same_day": same, "archetype": archetype,
                  "attention": level,
                  "withheld": edge is not None and edge < R.SUPPRESS_EDGE
                  if hasattr(R, "SUPPRESS_EDGE") else edge is not None and edge < -0.05})
    # 'Calmer than most' = the five highest chances of no big move. Where every
    # stock has its stored forecast probabilities for the day, those are used
    # (the same numbers as the chance column, no hindsight); otherwise the
    # bundled model's move score.
    stored = {t: r.get("t6_probs") for t, r in D["rows"].items()}
    if all(p and p.get("neutral") is not None for p in stored.values()):
        p_move = pd.Series({t: 1 - float(p["neutral"]) for t, p in stored.items()})
        rank = p_move.rank(method="first").astype(int)
        for t, o in looks.items():
            o.update({"p_move_model": o["p_move"], "activity_model": o["activity"], "p_move": float(p_move[t]),
                      "move_rank": int(rank[t]), "basis": "stored",
                      "activity": "quieter than most" if int(rank[t]) <= O.N_QUIET else "usual"})
        D["calm_source"] = "stored"
    else:
        D["calm_source"] = "model"
    D["outlook"] = looks
    D["t5_threshold"] = t5_thr
    D["t6_threshold"] = float(art.get("return_threshold", 1.0))
    D["evidence"] = ev
    D["art_mean"] = O.average_day(art).to_dict()
    return D


# ═══════════════════════════════════════════════════════════════════════════
# SNAPSHOTS  (a live date frozen into the package, so the sample can be rebuilt
# offline from real database inputs: signal-report --date D --save-snapshot)
# ═══════════════════════════════════════════════════════════════════════════

def snapshot_path(root, date):
    return Path(root) / "tests" / "fixtures" / f"live_snapshot_{date}.json"


def _jsonable(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, set):
        return sorted(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def save_snapshot(root, D, X, returns, model):
    """The live inputs exactly as gathered (before assembly), the 21 features
    per stock and the last 70 daily returns."""
    path = snapshot_path(root, D["date"])
    blob = {"saved_at": datetime.now(SGT).isoformat(timespec="minutes"), "model": model,
            "D": D, "X": X.to_dict(orient="index"),
            "returns": returns.tail(70).to_json(orient="split", date_format="iso")}
    path.write_text(json.dumps(blob, default=_jsonable, indent=1))
    return path


def gather_snapshot(root, date):
    import io
    root = Path(root)
    blob = json.loads(snapshot_path(root, date).read_text())
    with open(root / "models" / "task6_artifacts_v1.pkl", "rb") as fh:
        art = pickle.load(fh)
    D = blob["D"]
    D["mode"], D["source"] = "sample", "snapshot"
    D["narrative_saved"] = blob.get("narrative")
    for r in D["rows"].values():                       # same 30-day headline window as the live path
        r["headlines"] = pick_headlines(r.get("headlines") or [], date)
    X = pd.DataFrame.from_dict(blob["X"], orient="index")[art["feature_names"]].astype(float)
    returns = pd.read_json(io.StringIO(blob["returns"]), orient="split")
    returns.index = pd.to_datetime(returns.index).tz_localize(None)
    ev = O.load_evidence(root)
    D["findings"] = research_findings(root, date)
    D["provenance"] = [f"Inputs: a saved copy of the project database for {date} "
                       f"(tests/fixtures/live_snapshot_{date}.json, saved {blob['saved_at']} by signal-report "
                       "--save-snapshot): Task 1-6 stored outputs, today's headlines and Task 5 summaries, "
                       "adjusted daily closes."] + list(D.get("provenance") or [])
    return assemble(D, art, blob["model"], X, returns, ev)


def research_findings(root, date):
    """Aggregate results of the research programme, read from the bundled
    research outputs; the forecast record counts only forecasts whose
    three-day outcome had matured by the report date."""
    res = Path(root) / "outputs" / "research"
    f = {}
    r4 = pd.read_csv(res / "R4_t5_replay.csv")
    row = r4[r4.variant == "net"].iloc[0]
    f["r4_ic"], f["r4_days"] = float(row.rank_IC_1d), int(row.IC_days)
    r7 = pd.read_csv(res / "R7_t5_rule_audit.csv")
    f["event_fired"] = float(r7[r7.factor.str.startswith("Bullish corporate events")]["fired%"].sum())
    r8 = pd.read_csv(res / "R8_rl_v8_replay.csv").set_index("variant")
    f["pos_v7"], f["pos_v8"] = float(r8.loc["rl_v7", "bull%"]), float(r8.loc["rl_v8", "bull%"])
    f["ic_v7"], f["ic_v8"] = float(r8.loc["rl_v7", "rank_IC_1d"]), float(r8.loc["rl_v8", "rank_IC_1d"])
    r2 = pd.read_csv(res / "R2_horizon_calibration.csv")
    x = r2[(r2.h == 3) & (r2.calib == "oof")]
    f["oof_lo"], f["oof_hi"], f["oof_neutral"] = float(x.acc.min()), float(x.acc.max()), float(x.neutral_acc.iloc[0])
    r3 = pd.read_csv(res / "R3_lookback_margin.csv")
    f["sw_lo"], f["sw_hi"], f["sw_n"] = float(r3.acc.min()), float(r3.acc.max()), int(len(r3))
    f["sw_sw_lo"], f["sw_sw_hi"] = int(r3.swaps.min()), int(r3.swaps.max())
    r13 = pd.read_csv(res / "R13_champion_consistency.csv")
    f["r13_leads"], f["r13_n"] = int(r13.winner_leads_f1.sum()), int(len(r13))
    r6 = pd.read_csv(res / "R6_long_history.csv")
    y = r6[(r6.signal == "REV1") & (r6.h == 1) & r6.period.str.startswith("2019")].iloc[0]
    f["rev_ic"], f["rev_t"] = float(y.IC), float(y.IC_t)
    r10 = pd.read_csv(res / "R10_suppression_replay.csv")
    m = r10[pd.to_datetime(r10.date).map(lambda d: d + 3 * SESSION) <= pd.Timestamp(date)]
    hit = m.pred == m.actual
    f.update({"n": int(len(m)), "acc": float(hit.mean()), "neutral": float((m.actual == "neutral").mean()),
              "shown": float(hit[m.decision == "shown"].mean()), "hidden": float(hit[m.decision == "withheld"].mean()),
              "ge90_n": int((m.p >= 0.9).sum()), "ge90_acc": float(hit[m.p >= 0.9].mean()),
              "first": str(m.date.min()), "last": str(m.date.max())})
    return f


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE INPUTS (stored project records only)
# ═══════════════════════════════════════════════════════════════════════════

def gather_sample(root, date=SAMPLE_DATE, period_sessions=PERIOD_SESSIONS):
    root = Path(root)
    if snapshot_path(root, date).exists():
        return gather_snapshot(root, date)
    n_s = int(period_sessions)
    names, sectors = names_and_sectors(root)
    with open(root / "models" / "task6_artifacts_v1.pkl", "rb") as fh:
        art = pickle.load(fh)
    model = art.get("best_model_name", "RandomF")
    res = root / "outputs" / "research"

    fx_all = pd.read_csv(root / "tests" / "fixtures" / "feature_rows.csv")
    stored_dates = sorted(fx_all.date.unique())
    if date not in stored_dates:
        raise ValueError(f"The offline sample needs stored Task 1-4 feature rows: {', '.join(stored_dates)}")
    fx = fx_all[fx_all.date == date].set_index("ticker")[art["feature_names"]]
    daily, n_filled = stored_daily_returns(root)
    returns = daily[daily.index <= date]
    if date not in returns.index:
        raise ValueError(f"No stored close-to-close return for {date}")
    today = returns.loc[date]

    # The outlook record as it was known at the report-date close: R14 on the
    # forecast dates whose three-session outcome had matured by then.
    ev = O.load_evidence(root)
    if pd.Timestamp(ev["window"].split(" to ")[-1]) + O.HORIZON * SESSION > pd.Timestamp(date):
        from .research import r14_outlook
        ev = r14_outlook.run(root, as_of=date, write=False)

    # Period: the n_s sessions ending on the report date (data to its close).
    start = (pd.Timestamp(date) - n_s * SESSION).strftime("%Y-%m-%d")
    first = (pd.Timestamp(start) + SESSION).strftime("%Y-%m-%d")
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(first, date, freq=SESSION)]
    block = returns.reindex(days)
    wf = pd.read_csv(res / "walk_forward_predictions.csv")
    stored = wf[(wf.h == n_s) & (wf.calib == "oof") & (wf.model == "RandomF") & (wf.date == start)]
    if len(stored):
        period_ret = stored.set_index("ticker").ret
    elif not block.isna().any().any():
        period_ret = ((1 + block / 100).prod() - 1) * 100
    else:
        raise ValueError(f"No stored {n_s}-session return for the period ending {date}")

    r8 = pd.read_csv(res / "R8_rl_v8_readings.csv")
    # One evidence reading per trading day of the period (a reading stored on
    # a weekend is not shown).
    week = r8[r8.date.isin(days)]
    r8d = r8[r8.date == date].set_index("ticker")
    # Fired Task 5 rules: the production rule layer re-run on the stored
    # feature rows, kept only where it reproduces the stored rl_v8 net score
    # (research/r8_rule_detail.py). The live report reads them from the audit table.
    detail_path = res / "R8_rl_v8_rule_detail.csv"
    detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame(
        columns=["ticker", "date", "reproduced", "factor", "direction", "source", "weight", "decisive"])
    detail = detail[(detail.date == date) & (detail.reproduced.astype(bool)) & detail.factor.notna()]
    signals = {t: [{"factor": s.factor, "direction": s.direction, "source": s.source,
                    "weight": float(s.weight), "decisive": bool(s.decisive)} for s in g.itertuples()]
               for t, g in detail.groupby("ticker")}
    # Record and withholding use only what was known at the report-date close:
    # a three-session forecast made on day d matured three sessions later.
    r10 = pd.read_csv(res / "R10_suppression_replay.csv")
    horizon = int(art.get("forward_horizon", O.HORIZON))
    matured_on = pd.to_datetime(r10.date).map(lambda d: d + horizon * SESSION).dt.strftime("%Y-%m-%d")
    rec = r10[matured_on <= date]
    # The stored live decision for the report date (forecast, probability and
    # each stock's record as the live system saw it), else the last one before.
    rec_date = date if (r10.date == date).any() else r10[r10.date < date].date.max()
    r10d = r10[r10.date == rec_date].set_index("ticker")
    live_today = rec_date == date

    rows = {}
    for t in fx.index:
        wk = week[week.ticker == t]
        rd = wk.set_index("date")
        mv = {d: (None if pd.isna(block.loc[d, t]) else float(block.loc[d, t])) for d in days}
        per_day = {d: {"reading": rd.call_rl_v8.get(d) if d in rd.index else None,
                       "event": bool(rd.evt_fired_v8.get(d)) if d in rd.index else False,
                       "move": mv[d]} for d in days}
        rows[t] = {
            "ticker": t, "name": names[t], "sector": sectors[t],
            "move": float(today[t]), "period_ret": float(period_ret.get(t, np.nan)),
            "daily_moves": [[d, v] for d, v in mv.items() if v is not None], "days": per_day,
            "sentiment": r8d.loc[t, "call_rl_v8"], "net": float(r8d.loc[t, "net_rl_v8"]),
            "week_pos": int((wk.call_rl_v8 == "bullish").sum()),
            "week_neg": int((wk.call_rl_v8 == "bearish").sum()),
            "week_n": int(len(wk)),
            "edge": float(r10d.loc[t, "edge"]) if t in r10d.index and pd.notna(r10d.loc[t, "edge"]) else None,
            "n_prior": int(r10d.loc[t, "n_prior"]) if t in r10d.index else 0,
            "t6_class": r10d.loc[t, "pred"] if live_today and t in r10d.index else None,
            "t6_prob": float(r10d.loc[t, "p"]) if live_today and t in r10d.index else None,
            "t6_source": "stored live forecast" if live_today and t in r10d.index else "bundled model",
            "features": fx.loc[t].to_dict(), "summary": None, "drivers": [], "risks": [],
            "headlines": [], "event_types": [], "warning": None, "signals": signals.get(t, []),
        }
    as_of = ev.get("as_of")
    D = {"date": date, "mode": "sample", "names": names, "sectors": sectors, "rows": rows,
         "period": {"start": start, "first": first, "end": date, "sessions": n_s, "days": days,
                    "reading_dates": sorted(week.date.unique().tolist())},
         "record": {"as_of": date, "n": int(len(rec)), "last_forecast": rec.date.max(),
                    "accuracy": float((rec.pred == rec.actual).mean()),
                    "always_neutral": float((rec.actual == "neutral").mean()),
                    "shown_acc": float((rec[rec.decision == "shown"].pred == rec[rec.decision == "shown"].actual).mean()),
                    "withheld_acc": float((rec[rec.decision == "withheld"].pred == rec[rec.decision == "withheld"].actual).mean()),
                    "source": "stored live champion forecasts matured by the report date (R10)"},
         "model": {"name": model, "trained_at": str(art.get("trained_at", ""))[:10],
                   "train_window": art.get("train_window"),
                   "in_training_window": O.in_training_window(date, art)},
         "findings": research_findings(root, date),
         "versions": {"rule_layer": "rl_v8 (replay of the stored inputs)", "patterns": "11 patterns",
                      "renderer": "deterministic (no language-model call in the sample)"},
         "inputs_today": {"articles": int(fx.n_articles.sum()), "political": int(fx.n_political.iloc[0]),
                          "vix": float(fx.vix.iloc[0]), "pc_ratio": float(fx.pc_vol_ratio.iloc[0]),
                          "options_bearish": bool(fx.options_bearish.iloc[0]),
                          "regime": int(fx.regime_encoded.iloc[0])},
         "provenance": [
             f"Inputs: stored Task 1-4 feature rows for {pd.Timestamp(date):%d %b} (tests/fixtures/feature_rows.csv, "
             f"stored for {' and '.join(f'{pd.Timestamp(x):%d %b}' for x in stored_dates)}) and rl_v8 readings for "
             "the period (R8_rl_v8_readings.csv)." + (SAMPLE_DATE_NOTE if date == RECORDS_DATE else ""),
             "Prices: stored walk-forward returns by session; sessions without a stored one-day return are filled "
             f"exactly from stored three- or five-session returns ({n_filled} filled). Today's move and the period "
             "change use closes up to the report date only.",
             f"Forecasts: the stored live champion forecast for {rec_date} and each stock's record as the live "
             f"system saw it (R10); the record quoted counts the {len(rec)} forecasts matured by {date}. 'Calmer "
             "than most': bundled Task 6 model (inside its training period).",
             (f"Outlook record: R14 restricted to the {ev['n_dates']} walk-forward test dates ({ev['window']}) whose "
              f"three-session outcome had matured by {date}; the frozen record (31 dates to 15 Sep) is in the final "
              "report." if as_of else
              f"Outlook record: R14, {ev['n_dates']} walk-forward test dates ({ev['window']}), whose three-session "
              f"outcomes had all matured by {date}."),
             f"Signals (pages 1-2): the production Task 5 rule layer re-run on these feature rows, reproducing the "
             f"stored rl_v8 score for {len(signals)} of 15 stocks (R8_rl_v8_rule_detail.csv); the live report reads "
             "them from the audit table. Week chart: rl_v8 readings and event flags (R8). Not in the sample: Task 5 "
             "summaries and headlines (database only).",
         ]}
    return assemble(D, art, model, fx, returns, ev)


# ═══════════════════════════════════════════════════════════════════════════
# LIVE INPUTS (project database)
# ═══════════════════════════════════════════════════════════════════════════

def gather_live(ns, cfg, report_date=None, save=False):
    n_s = int(cfg["t7"].get("period_sessions", PERIOD_SESSIONS))
    """Inputs from the project database and adjusted closing prices.

    ns holds SUPABASE_ENGINE, SGX_STOCKS, extract_features_row (task6_features)
    and get_current_champion (task6_runtime). Today's per-stock record comes
    from task7_report.build_report_input, so this report and the Research
    Report read the same input.
    """
    from sqlalchemy import text
    import task7_report as R
    engine = ns["SUPABASE_ENGINE"]

    def q(sql, **p):
        with engine.connect() as conn:
            return conn.execute(text(sql), p).fetchall()

    if report_date is None:
        r = q("""SELECT a.d::text FROM
                   (SELECT analysis_date::date d, COUNT(DISTINCT ticker) n FROM llm_analyses GROUP BY 1) a
                 JOIN (SELECT prediction_date d, COUNT(DISTINCT ticker) n FROM task6_predictions
                       WHERE is_champion GROUP BY 1) p ON a.d = p.d
                 WHERE a.n >= 15 AND p.n >= 15 ORDER BY a.d DESC LIMIT 1""")
        report_date = r[0][0] if r else None
    if not report_date:
        raise RuntimeError("No date has all 15 stocks in both Task 5 and Task 6")

    from .configuration import ROOT
    names, sectors = names_and_sectors(ROOT)
    with open(cfg["paths"]["t6_artifacts"], "rb") as fh:
        art = pickle.load(fh)
    model = ns["get_current_champion"]() if "get_current_champion" in ns else art.get("best_model_name")
    ev = O.load_evidence(ROOT)

    px = R.load_price_panel(list(names))
    px = px[px.index <= pd.Timestamp(report_date)]
    returns = px.pct_change(fill_method=None) * 100
    if returns.index[-1].strftime("%Y-%m-%d") != report_date:
        raise RuntimeError(f"No closing prices for {report_date}")
    start_ts = returns.index[-n_s - 1]
    start = start_ts.strftime("%Y-%m-%d")
    period_ret = (px.iloc[-1] / px.loc[start_ts] - 1) * 100

    week = q("""SELECT ticker, analysis_date::date::text, sentiment FROM llm_analyses
                WHERE analysis_date::date > CAST(:s AS DATE) AND analysis_date::date <= CAST(:e AS DATE)""",
             s=start, e=report_date)
    days = [d.strftime("%Y-%m-%d") for d in returns.index[-n_s:]]
    week = [w for w in week if w[1] in days]          # one reading per trading day
    # Days on which the rule layer saw more corporate-event news than usual
    # (the "Unusual ... corporate event activity" rules in the audit record).
    events = {}
    try:
        for t, d, fr in q("""SELECT ticker, analysis_date::date::text, features_result
                             FROM llm_analyses_audit
                             WHERE analysis_date::date > CAST(:s AS DATE)
                               AND analysis_date::date <= CAST(:e AS DATE) ORDER BY id""",
                          s=start, e=report_date):
            fr = fr if isinstance(fr, dict) else json.loads(fr or "{}")
            events[(t, d)] = any("corporate event" in str(f.get("factor", "")).lower()
                                 for f in (fr or {}).get("features", []))
    except Exception as e:                                       # noqa: BLE001
        print(f"  note: event lookup unavailable ({e!r})")
    # Today's company articles (the ones counted in "News today"), with their
    # FinBERT tone: the strongest positive and negative headline are shown.
    heads, ev_types = {}, {}
    try:
        for t, title, src, at, sent, score, flag in q("""
                SELECT ticker, title, source, published::text, sentiment, sentiment_score, event_flag
                FROM news_articles
                WHERE CAST(date AS TEXT) = :e AND category = 'company' AND title IS NOT NULL""",
                                                           e=report_date):
            heads.setdefault(t, []).append({"title": title, "source": src, "at": at,
                                            "sentiment": sent, "score": float(score or 0)})
            if flag and flag != "none":
                ev_types.setdefault(t, set()).add(flag)
    except Exception as e:                                       # noqa: BLE001
        print(f"  note: headline lookup unavailable ({e!r})")

    rec = q("""SELECT COUNT(*), AVG(CASE WHEN predicted_class = actual_class THEN 1.0 ELSE 0.0 END),
                      AVG(CASE WHEN actual_class = 'neutral' THEN 1.0 ELSE 0.0 END)
               FROM task6_predictions WHERE is_champion AND actual_class IS NOT NULL
                 AND forecast_target <= CAST(:d AS DATE)""", d=report_date)[0]

    from task7_generate import generate_and_store
    rows, feats = {}, {}
    for t in names:
        res = generate_and_store(engine, t, report_date, groq_api_key=None, store=False, verbose=False)
        f = ns["extract_features_row"](t, report_date)
        if not res or "input" not in res or f is None:
            raise RuntimeError(f"{t}: incomplete stored inputs on {report_date}")
        inp = res["input"]
        att = res.get("attribution") or {}
        feats[t] = f
        t5, t6, track = inp["t5"], inp["t6"], inp["reliability"]
        wk = [s for (tk, d, s) in week if tk == t]
        rmap = {d: s for (tk, d, s) in week if tk == t}
        dm = returns[t].iloc[-n_s:]
        per_day = {d.strftime("%Y-%m-%d"): {"reading": rmap.get(d.strftime("%Y-%m-%d")),
                                            "event": events.get((t, d.strftime("%Y-%m-%d")), False),
                                            "move": None if pd.isna(v) else float(v)}
                   for d, v in dm.items()}
        rows[t] = {
            "ticker": t, "name": names[t], "sector": sectors[t],
            "move": float(returns[t].iloc[-1]), "period_ret": float(period_ret[t]),
            "daily_moves": [[d.strftime("%Y-%m-%d"), float(v)] for d, v in dm.items()],
            "days": per_day,
            "sentiment": t5["sentiment"], "net": att.get("net_score"),
            # Fired rules as the rule layer computed them (llm_analyses_audit),
            # not the LLM-written key_drivers (see task7_generate._sec_key_evidence).
            "signals": [{"factor": x.get("factor"), "direction": x.get("direction"),
                         "source": x.get("source"), "weight": x.get("weight"),
                         "decisive": x.get("decisive")}
                        for x in att.get("features") or []],
            "week_pos": wk.count("bullish"), "week_neg": wk.count("bearish"), "week_n": len(wk),
            "edge": track.get("t6_edge"), "n_prior": track.get("t6_n", 0),
            "t6_class": t6["predicted_class"], "t6_prob": t6["probability"],
            "t6_probs": t6.get("probabilities"), "t6_model": t6.get("champion_model"),
            "t6_acc": track.get("t6_accuracy"), "t6_base": track.get("t6_baseline"),
            "t6_source": "stored live forecast",
            "features": f, "summary": t5.get("summary"),
            "drivers": _as_list(t5.get("key_drivers"))[:2], "risks": _as_list(t5.get("risk_factors"))[:1],
            "headlines": pick_headlines(heads.get(t, []), report_date), "event_types": sorted(ev_types.get(t, [])),
            "warning": t5.get("warning_level"),
        }
    X = pd.DataFrame(feats).T.reindex(columns=art["feature_names"]).astype(float)
    D = {"date": report_date, "mode": "live", "findings": research_findings(ROOT, report_date), "names": names, "sectors": sectors, "rows": rows,
         "period": {"start": start, "first": returns.index[-n_s].strftime("%Y-%m-%d"),
                    "end": report_date, "sessions": n_s, "days": days,
                    "reading_dates": sorted({d for (_, d, _) in week})},
         "record": {"as_of": report_date, "n": int(rec[0] or 0),
                    "accuracy": float(rec[1]) if rec[1] is not None else None,
                    "always_neutral": float(rec[2]) if rec[2] is not None else None,
                    "shown_acc": None, "withheld_acc": None,
                    "source": "stored champion forecasts matured by the report date"},
         "model": {"name": model, "trained_at": str(art.get("trained_at", ""))[:10],
                   "train_window": art.get("train_window"),
                   "in_training_window": O.in_training_window(report_date, art)},
         "versions": {"rule_layer": cfg["t5"]["rule_layer_version"], "patterns": "11 patterns",
                      "renderer": "Task 5 language-model summary where it passed its checks; all numbers by template"},
         "inputs_today": {"articles": int(X.n_articles.sum()), "political": int(X.n_political.iloc[0]),
                          "vix": float(X.vix.iloc[0]), "pc_ratio": float(X.pc_vol_ratio.iloc[0]),
                          "options_bearish": bool(X.options_bearish.iloc[0]),
                          "regime": int(X.regime_encoded.iloc[0])},
         "provenance": ["Project database: Tasks 1-6 stored outputs for the period; adjusted daily closes.",
                        "Record of each statement: research R14, R10 and the stored champion record."]}
    if save:
        print(f"Snapshot: {save_snapshot(ROOT, D, X, returns, model)}")
    return assemble(D, art, model, X, returns, ev)


# ═══════════════════════════════════════════════════════════════════════════
# WORDS
# ═══════════════════════════════════════════════════════════════════════════

LEAD_WORD = {"company news": "Company news", "political news": "Political news",
             "price indicators": "Price-chart signals", "market conditions": "Market conditions",
             "options market": "The options market"}
SECTOR_ORDER_NOTE = "sectors sorted by their average change"


def news_count(r):
    return int(round(float(r["features"].get("n_articles", 0) or 0)))


NET_LINE = 0.35          # Task 5: a reading is positive or negative beyond a net score of ±0.35
NEWS_TONE_LINE = 0.15    # Task 5 news rule: (positive - negative) / articles beyond ±0.15
TONE_WORD = {"positive": "Mostly positive", "negative": "Mostly negative", "mixed": "Mixed",
             "few": "Too few", "none": "None"}
EVENT_WORD = {"ma": "merger or acquisition", "earnings": "earnings", "dividend": "dividend",
              "upgrade": "upgrade", "downgrade": "downgrade", "warning": "profit warning",
              "layoff": "layoffs", "management": "management change"}
CHART_SHORT = {
    "RSI oversold — potential bounce": "Oversold", "RSI overbought — potential pullback": "Overbought",
    "MACD bullish crossover": "Trend turned up", "MACD bearish crossover": "Trend turned down",
    "Price above Bollinger upper band": "Above range", "Price below Bollinger lower band": "Below range",
    "Price above VWAP": "Above avg price", "Weekly timeframe confirms bullish": "Weekly up",
    "Weekly timeframe confirms bearish": "Weekly down",
}
DECISIVE_WORD = {
    "Positive company news sentiment": "the positive news", "Negative company news sentiment": "the negative news",
    "Unusual bullish corporate event activity": "the extra event news",
    "Unusual bearish corporate event activity": "the extra warning news",
    "RSI oversold — potential bounce": "the 'oversold' signal",
    "RSI overbought — potential pullback": "the 'overbought' signal",
    "MACD bullish crossover": "the trend signal", "MACD bearish crossover": "the trend signal",
    "Price above Bollinger upper band": "the above-range signal",
    "Price below Bollinger lower band": "the below-range signal",
    "Price above VWAP": "the average-price signal",
    "Weekly timeframe confirms bullish": "the weekly trend", "Weekly timeframe confirms bearish": "the weekly trend",
}
REASON_WORD = {"company news": "its company news", "price indicators": "its price-chart signals",
               "market conditions": "market conditions, which the model reads as calmer than on an average day",
               "options market": "the options market", "political news": "the day's political news"}


def news_split(r):
    f, n = r["features"], news_count(r)
    pos = int(round(float(f.get("news_bullish_ratio", 0) or 0) * n))
    neg = int(round(float(f.get("news_bearish_ratio", 0) or 0) * n))
    return n, pos, neg, max(n - pos - neg, 0)


def news_tone(r):
    """The tone of today's company articles, on the Task 5 news rule's line."""
    f, n = r["features"], news_count(r)
    if n == 0:
        return "none"
    if n < 3:
        return "few"
    d = float(f.get("news_bullish_ratio", 0) or 0) - float(f.get("news_bearish_ratio", 0) or 0)
    return "positive" if d > NEWS_TONE_LINE else "negative" if d < -NEWS_TONE_LINE else "mixed"


def tone_label(r):
    """'Mixed' only when positive and negative articles are both present in number;
    a day of mostly neutral articles is called that."""
    tone = news_tone(r)
    if tone == "mixed":
        n, pos, neg, _ = news_split(r)
        if pos + neg < 0.2 * n:
            return "Mostly neutral"
    return TONE_WORD[tone]


def news_text(r):
    n, pos, neg, neu = news_split(r)
    if n == 0:
        return "No company articles were collected today."
    tone = news_tone(r)
    s = f"{n} article{'s' if n != 1 else ''} collected today: {pos} positive, {neg} negative, {neu} neutral"
    s += f" ({tone_label(r).lower()})." if tone in ("positive", "negative", "mixed") else ", too few to read."
    f = r["features"]
    nb, nr = int(f.get("n_bullish_events", 0) or 0), int(f.get("n_bearish_events", 0) or 0)
    if r.get("event_types"):
        s += " Event tags: " + ", ".join(EVENT_WORD.get(x, x) for x in r["event_types"]) + "."
    elif nb or nr:
        parts = ([f"{nb} about earnings, dividends, upgrades or deals"] if nb else []) + \
                ([f"{nr} about warnings, downgrades or layoffs"] if nr else [])
        s += " " + " and ".join(parts) + "."
    return s


def _clean_title(h):
    t = str(h.get("title") or "").strip()
    src = str(h.get("source") or "").strip()
    if src and t.lower().endswith(" - " + src.lower()):
        t = t[: -len(src) - 3]
    return t if len(t) <= 95 else t[:92].rstrip() + "…"


def headline_text(r):
    """Live report: the strongest positive and the strongest negative headline of the day."""
    out = []
    for sign, h in (("+", next((x for x in r.get("headlines") or [] if x.get("sentiment") == "bullish"), None)),
                    ("−", next((x for x in r.get("headlines") or [] if x.get("sentiment") == "bearish"), None))):
        if h:
            at = pd.to_datetime(str(h.get("at") or ""), errors="coerce", utc=True)
            src = ", ".join(x for x in (h.get("source"), None if pd.isna(at) else f"{at:%d %b}") if x)
            out.append(f"{sign} “{_clean_title(h)}” ({src})")
    return "   ".join(out) or None


def pick_headlines(items, report_date=None, days=30):
    """Strongest positive and negative headline published in the news look-back
    window (30 days, settings/solution.json); the news search also returns
    older articles, which are not shown."""
    if report_date:
        end = pd.Timestamp(report_date, tz="UTC") + pd.Timedelta(days=1)
        def recent(h):
            at = pd.to_datetime(str(h.get("at") or ""), errors="coerce", utc=True)
            return not pd.isna(at) and end - pd.Timedelta(days=days + 1) <= at <= end
        items = [h for h in items if recent(h)]
    items = [h for h in items if summary_is_clean(str(h.get("title") or ""))]   # no quoted advice or targets
    pos = [h for h in items if h.get("sentiment") == "bullish"]
    neg = [h for h in items if h.get("sentiment") == "bearish"]
    return ([max(pos, key=lambda h: h.get("score") or 0)] if pos else []) + \
           ([max(neg, key=lambda h: h.get("score") or 0)] if neg else [])


def chart_short(r):
    fired = [CHART_SHORT[_base_factor(s["factor"])] for s in card_signals(r.get("signals"))
             if s.get("source") in CHART_SOURCES and _base_factor(s["factor"]) in CHART_SHORT]
    return " · ".join(dict.fromkeys(fired).keys()) or "—"


def chart_text(r):
    parts = []
    rsi = r["features"].get("rsi")
    if rsi is not None and not np.isnan(float(rsi)):
        rsi = float(rsi)
        parts.append(f"RSI {rsi:.0f}, " + ("below 30: the price fell fast ('oversold')" if rsi < 30 else
                                          "above 70: the price rose fast ('overbought')" if rsi > 70 else
                                          "between 30 and 70: no unusually fast move"))
    parts += [plain_signal(s["factor"]) for s in card_signals(r.get("signals"))
              if s.get("source") in CHART_SOURCES and "RSI" not in str(s.get("factor"))]
    if not parts:
        return "No price-chart signal."
    out = "; ".join(parts)
    return out[:1].upper() + out[1:] + "."


def _decisive(r):
    """Shown stock-specific rules without which the reading would be neutral
    (Task 5's flips_call; recomputed from the net score where not stored)."""
    if r["sentiment"] == "neutral" or r.get("net") is None:
        return []
    net, out = float(r["net"]), []
    for s in card_signals(r.get("signals")):
        if s.get("source") in MARKET_SOURCES:
            continue
        dec = s.get("decisive")
        if dec is None:
            rest = net - float(s.get("weight") or 0) * (1 if s.get("direction") == "bullish" else -1)
            dec = (net > NET_LINE) != (rest > NET_LINE) or (net < -NET_LINE) != (rest < -NET_LINE)
        if dec:
            out.append(DECISIVE_WORD.get(_base_factor(s["factor"]), f"'{short_signal(s['factor'])}'"))
    return list(dict.fromkeys(out))


def political_net(r):
    return sum(float(x.get("weight") or 0) * (1 if x.get("direction") == "bullish" else -1)
               for x in (r.get("signals") or []) if x.get("source") == "task1_political")


def reading_text(r, D):
    others = [c["reading"] for d, c in r["days"].items() if d != D["date"] and c.get("reading")]
    s = f"{EVIDENCE_WORD[r['sentiment']]} today"
    if r.get("net") is not None:
        s += f" (score {pts(float(r['net']))}; positive above +0.35, negative below −0.35)"
    if others:
        same, k = sum(x == r["sentiment"] for x in others), len(others)
        days = f"{k} earlier recorded day{'s' if k != 1 else ''}"
        s += (f", and on all {days}" if same == k and k > 1 else f", but on none of the {days}" if same == 0
              else f", and on {same} of the {days}")
    s += "."
    stock = [x for x in card_signals(r.get("signals")) if x.get("source") not in MARKET_SOURCES]
    if r["sentiment"] != "neutral":
        dec = _decisive(r)
        if len(dec) == 1:
            s += f" Without {dec[0]} it would be neutral."
        elif len(dec) == 2:
            s += f" It needs both {dec[0]} and {dec[1]}: without either it would be neutral."
        elif dec:
            s += f" It needs all of {_names(dec)}: without any one it would be neutral."
        else:
            s += " No single rule decides it."
        if r.get("net") is not None and abs(float(r["net"]) - political_net(r)) <= NET_LINE:
            s += (f" Borderline: without the two political-news rules (unreliable, section 5) the score would be "
                  f"{pts(float(r['net']) - political_net(r))}, inside the neutral band.")
    elif not stock:
        s += " No stock-specific rule fired."
    else:
        s += " The rules for and against did not add up to a clear reading."
    if r.get("warning") == "high":
        s += f" Also, {WARNING_WORDS['high']}."
    return s


def explain_verdict(r):
    """Does the company news explain the move over the period, in one phrase."""
    ret, tone = r["period_ret"], news_tone(r)
    if abs(ret) < 1:
        return "barely moved"
    if tone in ("none", "few"):
        return "not enough news"
    if tone == "mixed":
        return "not clearly"
    return "same direction" if (tone == "positive") == (ret > 0) else "no"


def market_avg(D):
    return float(np.nanmean([r["period_ret"] for r in D["rows"].values()]))


def context(r, D):
    """The period move against the company news, the sector peers and the 15-stock
    average, and the conclusion the report draws from them (descriptive only)."""
    ret, mkt = r["period_ret"], market_avg(D)
    peers = [x for x in D["rows"].values() if x["sector"] == r["sector"] and x["ticker"] != r["ticker"]]
    pav = float(np.mean([p["period_ret"] for p in peers])) if peers else None
    verdict = explain_verdict(r)
    sector_wide = pav is not None and np.sign(pav) == np.sign(ret) and abs(pav) >= 0.5 * abs(ret)
    market_wide = np.sign(mkt) == np.sign(ret) and abs(mkt) >= 0.5 * abs(ret)
    wide = "sector-wide" if sector_wide else "market-wide" if market_wide else None
    against = bool(np.sign(ret) != np.sign(mkt) and abs(mkt) >= 0.2)
    if verdict == "barely moved":
        conclusion = "it barely moved over the period"
    elif verdict == "same direction":
        conclusion = "the company news and the price point the same way" + (f", and the move looks {wide}" if wide else "")
    elif verdict == "no":
        conclusion = "the company news and the price disagree" + (f"; the move looks {wide}" if wide else "")
    elif wide:
        conclusion = f"there is no company story in the news, and the move looks {wide}"
    else:
        conclusion = ("there is no company story in the news, and the stock moved "
                      + ("against the market" if against else "more than the market")
                      + "; sector or wider news, outside this report's sources, is the place to look")
    return {"verdict": verdict, "peers": peers, "peers_avg": pav, "market_avg": mkt, "wide": wide,
            "against": against, "conclusion": conclusion}


def context_text(r, D):
    """Card line 'Last 5 days in context'."""
    c, ret = context(r, D), r["period_ret"]
    k, p, n, _ = news_split(r)
    tone = news_tone(r)
    if tone in ("none", "few"):
        news = f"Too little company news to judge ({k} article{'s' if k != 1 else ''} collected today)"
    else:
        news = f"Company news collected today was {tone_label(r).lower()}"
    went = "fell" if ret < 0 else "rose"
    move = f"the price {went} {abs(ret):.1f}% over {_period_short(D)}"
    if c["peers"]:
        peer = (f"its sector peer {short(c['peers'][0]['name'])} {pct(c['peers'][0]['period_ret'])}"
                if len(c["peers"]) == 1 else f"its {len(c['peers'])} sector peers {pct(c['peers_avg'])} on average")
        move += f", {peer}, and the 15 stocks {pct(c['market_avg'])} on average"
    else:
        move += f", against {pct(c['market_avg'])} for the 15 stocks on average"
    concl = c["conclusion"]
    return f"{news}; {move}. So {concl}."


def today_text(r, D):
    """Card line 'Today: news vs price': the pattern engine's verdict in plain words."""
    thr, m, s, a = D.get("t5_threshold") or 0.6, r["move"], r["sentiment"], r["archetype"]
    if m is None:
        return "No same-day price available."
    moved = f"{'rose' if m > 0 else 'fell'} {abs(m):.1f}%" if abs(m) >= 0.05 else "was unchanged"
    band = f"a normal day's ±{thr:.1f}%"
    fc = FORECAST_WORD[r["t6_class"]].lower()
    if s == "neutral":
        t = f"No news signal today (neutral reading); the price {moved}"
        t += f", more than {band}: a move without a news signal." if abs(m) > thr else "."
        if a == "model_only":
            t += f" Task 6 still leans '{fc}'."
    else:
        word = EVIDENCE_WORD[s]
        if a in ("move_complete", "momentum", "mean_reversion"):
            t = f"{word} reading and the price {moved}, more than {band}: the news already shows in the price."
        elif a in ("dormant", "muted_pending", "muted_fading"):
            t = f"{word} reading, but the price {moved}, within {band}: the news has not shown up in the price."
        else:
            t = f"{word} reading, but the price {moved}: the price went against the news."
        t += {"momentum": " Task 6 expects the move to continue.", "mean_reversion": " Task 6 expects a pullback.",
              "move_complete": " Task 6 expects no big move from here.", "dormant": " Task 6 expects no big move.",
              "muted_pending": " Task 6 still expects that direction.", "muted_fading": " Task 6 expects the opposite.",
              "lag": " Task 6 sides with the news.", "contradicted": " Task 6 sides with the price.",
              "noise": " Task 6 has no clear view."}.get(a, "")
    return t + f" (Pattern: {PATTERN_NAME.get(a, a)}.)"


def next_text(r, o, calm_set):
    rng = o.get("range")
    s = (f"Normal 3-day swing ±{rng['half_width_pct']:.1f}% (±S${rng['half_width_pct'] * 10:.0f} on S$1,000)"
         if rng else "Normal 3-day swing not available")
    s += "; one of the five the model expects to be calmer." if r["ticker"] in calm_set else "."
    if r["withheld"]:
        s += " Up-or-down call hidden: on this stock the model has done worse than a simple rule."
    else:
        s += (f" The model's call ({FORECAST_WORD[r['t6_class']].lower()}) is shown: on this stock it has beaten "
              "a simple rule.")
    return s


def pts(x):
    return f"{x:+.2f}".replace("-", "−")


def rules_text(r):
    """Every rule Task 5 fired today, with its points (political rules summed)."""
    sig = r.get("signals") or []

    def name(s):
        n = SIGNAL_SHORT.get(_base_factor(s["factor"]), _base_factor(s["factor"])).replace("*", "")
        first = n.split()[0]
        return n if first.isupper() else n[:1].lower() + n[1:]

    def signed(s):
        return float(s.get("weight") or 0) * (1 if s.get("direction") == "bullish" else -1)
    shown = card_signals(sig)
    stock = [s for s in shown if s.get("source") not in MARKET_SOURCES]
    up = [f"{name(s)} {pts(signed(s))}" for s in stock if s.get("direction") == "bullish"]
    dn = [f"{name(s)} {pts(signed(s))}" for s in stock if s.get("direction") != "bullish"]
    mk = [f"{name(s)} {pts(signed(s))}" for s in shown if s.get("source") in MARKET_SOURCES]
    pol = sum(signed(s) for s in sig if s not in shown)
    if any(s not in shown for s in sig):
        mk.append(f"political news {pts(pol)} (unreliable, section 5)")
    parts = ([f"Up: {', '.join(up)}"] if up else []) + ([f"Down: {', '.join(dn)}"] if dn else []) + \
            ([f"Market-wide: {', '.join(mk)}"] if mk else [])
    return (". ".join(parts) + ".") if parts else "No rule fired."


REASON_SHORT = {"company news": "company news", "price indicators": "price-chart signals",
                "market conditions": "calm market conditions", "options market": "options market",
                "political news": "political news"}


def calm_short(o):
    lead, val = sorted(o["move_by_group"].items(), key=lambda kv: kv[1])[0]
    why = REASON_SHORT.get(lead, lead) if val < -0.005 else "no single reason"
    return f"{short(o['name'])} ({why}; recent swings {'agree' if o.get('vol_quiet') else 'larger'})"


def card_forecast(r, D=None):
    p = r.get("t6_prob")
    s = f"{FORECAST_WORD[r['t6_class']]} over the next 3 trading days"
    if p is not None:
        s += f" (chance {'>99' if p > 0.995 else f'{p * 100:.0f}'}%)"
    o = (D or {}).get("outlook", {}).get(r["ticker"]) if D else None
    if o and o.get("range"):
        s += f"; its normal 3-day swing is ±{o['range']['half_width_pct']:.1f}%"
    return s + (". Past record on this stock: about as good as a simple rule or better (✓), still only a weak "
                "signal." if not r["withheld"] else
                ". Past record on this stock: below a simple rule (✗), so this call carries no weight.")


def record_cell(r):
    edge = r.get("edge")
    if edge is None:
        return "too new"
    mark = "✗" if r["withheld"] else "✓"
    return f"{edge * 100:+.0f} pts {mark}".replace("+0 pts", "±0 pts").replace("-", "−")


def calm_reason(o):
    groups = sorted(o["move_by_group"].items(), key=lambda kv: kv[1])
    lead, val = groups[0]
    if val >= -0.005:
        return "No single input stood out."
    return f"Main reason: {REASON_WORD.get(lead, lead)}."


def model_reason(o):
    """Exact group Shapley values in words: the input family that lowered the
    model's chance of a move the most, by how many percentage points."""
    groups = sorted(o["move_by_group"].items(), key=lambda kv: kv[1])
    lead, val = groups[0]
    if val >= -0.005:
        return "no single input stood out; the model's chance of a move was low across the board."
    s = (f"{LEAD_WORD.get(lead, lead.capitalize())} lowered the model's chance of a move by "
         f"{abs(val) * 100:.0f} percentage points")
    if o.get("top_facts") and o.get("top_group") == lead:
        s += f": {o['top_facts'][0]['text']}"
    return s + "."


def largest_day(r):
    moves = [m for m in r["daily_moves"] if m[1] is not None and not np.isnan(m[1])]
    if not moves:
        return None
    return max(moves, key=lambda x: abs(x[1]))


def card_happened(r, D):
    ret = r["period_ret"]
    s = f"{'Rose' if ret > 0 else 'Fell'} {abs(ret):.1f}% over {_period_short(D)}"
    ld = largest_day(r)
    if ld and abs(ld[1]) >= max(1.5, 0.5 * abs(ret)) and np.sign(ld[1]) == np.sign(ret):
        s += (f", including {pct(ld[1])} on {pd.Timestamp(ld[0]):%a %d %b}" if abs(ld[1]) >= abs(ret)
              else f", most of it on {pd.Timestamp(ld[0]):%a %d %b} ({pct(ld[1])})" if abs(ld[1]) >= 0.6 * abs(ret)
              else f"; its biggest day was {pd.Timestamp(ld[0]):%a %d %b} ({pct(ld[1])})")
    elif ld:
        s += f"; its biggest day was {pd.Timestamp(ld[0]):%a %d %b} ({pct(ld[1])})"
    return s + f". Today {pct(r['move'])}."


# ═══════════════════════════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════════════════════════

EV_FILL = {"bullish": "#86b6ef", "bearish": "#f0a3a2", "neutral": "#e3e2dc", None: "#ffffff"}
UP, DOWN = "#2a78d6", "#e34948"


def _sector_groups(D):
    sect = {}
    for r in D["rows"].values():
        sect.setdefault(r["sector"], []).append(r)
    out = []
    for s, rs in sorted(sect.items(), key=lambda kv: -np.nanmean([r["period_ret"] for r in kv[1]])):
        out.append((s, float(np.nanmean([r["period_ret"] for r in rs])),
                    sorted(rs, key=lambda r: -r["period_ret"])))
    return out


def chart_week(D, path):
    """The week by sector: one square per trading day coloured by that day's
    evidence reading (star = more company-event news than usual), and a bar
    for the change over the week."""
    from matplotlib.patches import FancyBboxPatch
    days = D["period"]["days"]
    groups = _sector_groups(D)
    lines = []
    for s, avg, rs in groups:
        lines.append(("sector", s, avg))
        lines += [("stock", r, None) for r in rs]
    n = len(lines)
    fig = plt.figure(figsize=(7.0, 0.150 * n + 0.75))
    ax = fig.add_axes([0.0, 0.11, 1.0, 0.86])
    x_lab, x_day0, cw = 0.0, 3.2, 0.62
    x_bar0 = x_day0 + len(days) * cw + 0.55
    span = max(abs(r["period_ret"]) for r in D["rows"].values()) or 1
    scale = 2.3 / span
    zero = x_bar0 + 2.45
    for i, (kind, obj, avg) in enumerate(lines):
        y = n - 1 - i
        if kind == "sector":
            ax.text(x_lab, y + 0.42, obj, fontsize=7.6, fontweight="bold", color="#1a2e55", va="center")
            ax.text(x_day0 - 0.12, y + 0.42, f"avg {avg:+.1f}%", fontsize=6.8, color=INK2, va="center",
                    ha="right")
            ax.plot([x_lab, zero + 2.7], [y + 0.02, y + 0.02], color=GRID, lw=0.6)
            continue
        r = obj
        ax.text(x_lab + 0.12, y + 0.42, short(r["name"]), fontsize=7.4, color="#0b0b0b", va="center")
        for j, d in enumerate(days):
            c = r["days"].get(d, {})
            fill = EV_FILL.get(c.get("reading"), "#ffffff")
            ax.add_patch(FancyBboxPatch((x_day0 + j * cw + 0.06, y + 0.12), cw - 0.12, 0.62,
                                        boxstyle="round,pad=0,rounding_size=0.08", facecolor=fill,
                                        edgecolor="#c3c2b7" if c.get("reading") is None else fill, lw=0.6))
            if c.get("event"):
                ax.text(x_day0 + j * cw + cw / 2, y + 0.42, "★", ha="center", va="center", fontsize=7.4,
                        color="#0b0b0b")
        v = r["period_ret"]
        ax.add_patch(FancyBboxPatch((min(zero, zero + v * scale), y + 0.18), abs(v * scale), 0.5,
                                    boxstyle="round,pad=0,rounding_size=0.05", facecolor=UP if v > 0 else DOWN,
                                    edgecolor="none"))
        ax.text(zero + v * scale + (0.08 if v >= 0 else -0.08), y + 0.42, f"{v:+.1f}%", fontsize=7,
                color=INK2, va="center", ha="left" if v >= 0 else "right")
    ax.plot([zero, zero], [0, n - 0.1], color="#898781", lw=0.7)
    for j, d in enumerate(days):
        ax.text(x_day0 + j * cw + cw / 2, n + 0.25, f"{pd.Timestamp(d):%a}\n{pd.Timestamp(d):%d %b}",
                ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.text(zero, n + 0.25, f"Change over {_period_short(D)}", ha="center", va="bottom", fontsize=6.8,
            color=INK2)
    ax.set_xlim(-0.05, zero + 2.9)
    ax.set_ylim(-0.1, n + 1.1)
    ax.axis("off")
    # legend
    lx, ly = 0.01, 0.035
    for k, (label, key) in enumerate((("positive", "bullish"), ("neutral", "neutral"),
                                      ("negative", "bearish"), ("no reading", None))):
        fig.patches.append(FancyBboxPatch((lx + k * 0.115, ly), 0.018, 0.028, transform=fig.transFigure,
                                          boxstyle="round,pad=0,rounding_size=0.004", facecolor=EV_FILL[key],
                                          edgecolor="#c3c2b7" if key is None else EV_FILL[key], lw=0.6))
        fig.text(lx + k * 0.115 + 0.024, ly + 0.014, label, fontsize=6.8, color=INK2, va="center")
    fig.text(lx + 4 * 0.115 + 0.01, ly + 0.014, "★ more company-event news than usual (earnings, dividends, "
             "upgrades)", fontsize=6.8, color=INK2, va="center")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════

C_NAVY, C_BLUE, C_INK2, C_LINE = "1A2E55", "2A78D6", "52514E", "D9D8D3"
CHIP = {"bullish": ("DCEAFB", "1C5CAB"), "neutral": ("EFEEEA", "52514E"), "bearish": ("FBE1E0", "A12D2C")}
VERDICT = {"good": ("✓", "E4F3E4", "0A6E0A"), "warn": ("!", "FFF1D1", "8A5A00"),
           "bad": ("✗", "FBE1E0", "A12D2C"), "info": ("○", "EFEEEA", "52514E")}


class Doc:
    def __init__(self, footer=""):
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches, Cm
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        self.Pt, self.RGB, self.Inches, self.Cm, self.qn, self.El = Pt, RGBColor, Inches, Cm, qn, OxmlElement
        self.d = Document()
        for s in self.d.sections:
            s.page_width, s.page_height = Cm(21.0), Cm(29.7)
            s.left_margin = s.right_margin = Cm(1.8)
            s.top_margin, s.bottom_margin = Cm(1.5), Cm(1.5)
            s.footer_distance = Cm(0.8)
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.font.size = Pt(9.6)
        st.paragraph_format.space_after = Pt(3)
        st.paragraph_format.line_spacing = 1.08
        if footer:
            self._footer(footer)

    # ── low-level helpers ────────────────────────────────────────────────
    def _rgb(self, hexs):
        return self.RGB(int(hexs[0:2], 16), int(hexs[2:4], 16), int(hexs[4:6], 16))

    def _shade(self, cell, hexfill):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = self.El("w:shd")
        shd.set(self.qn("w:val"), "clear")
        shd.set(self.qn("w:color"), "auto")
        shd.set(self.qn("w:fill"), hexfill)
        tcPr.append(shd)

    def _cell_border(self, cell, **edges):
        tcPr = cell._tc.get_or_add_tcPr()
        b = self.El("w:tcBorders")
        for edge, (sz, color) in edges.items():
            e = self.El(f"w:{edge}")
            e.set(self.qn("w:val"), "single" if sz else "nil")
            e.set(self.qn("w:sz"), str(sz))
            e.set(self.qn("w:color"), color)
            b.append(e)
        tcPr.append(b)

    def _table_style(self, t, inside_h=True, outer=True, pad=(40, 70)):
        tblPr = t._tbl.tblPr
        borders = self.El("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = self.El(f"w:{edge}")
            on = (edge in ("top", "bottom") and outer) or (edge == "insideH" and inside_h)
            e.set(self.qn("w:val"), "single" if on else "nil")
            e.set(self.qn("w:sz"), "4")
            e.set(self.qn("w:color"), C_LINE)
            borders.append(e)
        tblPr.append(borders)
        mar = self.El("w:tblCellMar")
        for edge, v in (("top", pad[0]), ("bottom", pad[0]), ("left", pad[1]), ("right", pad[1])):
            e = self.El(f"w:{edge}")
            e.set(self.qn("w:w"), str(v))
            e.set(self.qn("w:type"), "dxa")
            mar.append(e)
        tblPr.append(mar)

    def _widths(self, t, widths):
        t.autofit = False
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = self.Inches(w)
        for i, w in enumerate(widths):
            t.columns[i].width = self.Inches(w)

    def _run(self, par, text, size=9.6, bold=False, color=None, italic=False):
        r = par.add_run(text)
        r.font.size, r.bold, r.italic = self.Pt(size), bold, italic
        if color:
            r.font.color.rgb = self._rgb(color)
        return r

    def _footer(self, text):
        p = self.d.sections[0].footer.paragraphs[0]
        self._run(p, text + "   ·   page ", size=7.5, color="898781")
        r = p.add_run()
        for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if kind:
                f = self.El("w:fldChar")
                f.set(self.qn("w:fldCharType"), kind)
                r._r.append(f)
            else:
                it = self.El("w:instrText")
                it.set(self.qn("xml:space"), "preserve")
                it.text = txt
                r._r.append(it)
        r.font.size = self.Pt(7.5)
        r.font.color.rgb = self._rgb("898781")

    def spacer(self, pt=2):
        p = self.d.add_paragraph()
        p.paragraph_format.space_after = self.Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run("")
        r.font.size = self.Pt(pt)
        return p

    # ── blocks ───────────────────────────────────────────────────────────
    def header_band(self, title, left_sub, right_lines):
        t = self.d.add_table(rows=1, cols=2)
        self._table_style(t, inside_h=False, outer=False, pad=(90, 140))
        a, b = t.rows[0].cells
        for c in (a, b):
            self._shade(c, C_NAVY)
        self._run(a.paragraphs[0], title, size=19, bold=True, color="FFFFFF")
        self._run(a.add_paragraph(), left_sub, size=9.5, color="CDE2FB")
        for k, line in enumerate(right_lines):
            par = b.paragraphs[0] if k == 0 else b.add_paragraph()
            par.alignment = 2
            self._run(par, line, size=8.6, color="FFFFFF" if k == 0 else "CDE2FB", bold=(k == 0))
        self._widths(t, [4.0, 2.9])
        self.spacer(3)

    def h(self, text, size=13, new_page=False, sub=None):
        p = self.d.add_paragraph()
        p.paragraph_format.page_break_before = new_page
        p.paragraph_format.space_before = self.Pt(0 if new_page else 8)
        p.paragraph_format.space_after = self.Pt(2)
        p.paragraph_format.keep_with_next = True
        self._run(p, text, size=size, bold=True, color=C_NAVY)
        pPr = p._p.get_or_add_pPr()
        bdr = self.El("w:pBdr")
        bot = self.El("w:bottom")
        for k, v in (("val", "single"), ("sz", "6"), ("space", "2"), ("color", "CDE2FB")):
            bot.set(self.qn(f"w:{k}"), v)
        bdr.append(bot)
        pPr.append(bdr)
        if sub:
            q = self.d.add_paragraph()
            q.paragraph_format.keep_with_next = True
            self._run(q, sub, size=8.8, color=C_INK2)
        return p

    def h2(self, text):
        p = self.d.add_paragraph()
        p.paragraph_format.space_before = self.Pt(6)
        p.paragraph_format.space_after = self.Pt(1)
        p.paragraph_format.keep_with_next = True
        self._run(p, text, size=10.5, bold=True, color=C_NAVY)

    def p(self, text, size=9.6, color=None, bold_lead=None, italic=False, after=3):
        par = self.d.add_paragraph()
        par.paragraph_format.space_after = self.Pt(after)
        if bold_lead:
            self._run(par, bold_lead, size=size, bold=True, color=C_NAVY)
        self._run(par, text, size=size, color=color, italic=italic)
        return par

    def bullet(self, text, lead=None, size=9.6):
        par = self.d.add_paragraph(style="List Bullet")
        par.paragraph_format.space_after = self.Pt(2)
        if lead:
            self._run(par, lead, size=size, bold=True, color=C_NAVY)
        self._run(par, text, size=size)

    def note(self, text, fill="FFF6E6", edge="EDA100", size=8.4):
        t = self.d.add_table(rows=1, cols=1)
        self._table_style(t, inside_h=False, outer=False, pad=(50, 110))
        c = t.cell(0, 0)
        self._shade(c, fill)
        self._cell_border(c, left=(18, edge))
        self._run(c.paragraphs[0], text, size=size, color="3D3C39")
        self._widths(t, [6.9])
        self.spacer(3)

    def tiles(self, items):
        """[(big, small, colour_hex)]"""
        t = self.d.add_table(rows=1, cols=len(items))
        self._table_style(t, inside_h=False, outer=False, pad=(70, 110))
        for i, (big, small, col) in enumerate(items):
            c = t.rows[0].cells[i]
            self._shade(c, "F3F6FA")
            self._cell_border(c, left=(4, "FFFFFF"), right=(4, "FFFFFF"))
            self._run(c.paragraphs[0], big, size=15, bold=True, color=col)
            self._run(c.add_paragraph(), small, size=7.8, color=C_INK2)
        self._widths(t, [6.9 / len(items)] * len(items))
        self.spacer(3)

    def chips_row(self, items, size=7.8):
        """[(verdict_key, text)] as one row of status chips (icon + label)."""
        t = self.d.add_table(rows=1, cols=len(items))
        self._table_style(t, inside_h=False, outer=False, pad=(35, 80))
        for i, (key, text) in enumerate(items):
            icon, fill, ink = VERDICT[key]
            c = t.rows[0].cells[i]
            self._shade(c, fill)
            self._cell_border(c, left=(6, "FFFFFF"), right=(6, "FFFFFF"))
            self._run(c.paragraphs[0], icon + "  ", size=8.4, bold=True, color=ink)
            self._run(c.paragraphs[0], text, size=size, color=ink)
        self._widths(t, [6.9 / len(items)] * len(items))
        self.spacer(3)

    def table(self, headers, rows, widths, font=8.2, chips=None, colors=None, align_right=(), bold_col0=False,
              pad=(40, 70)):
        """rows of strings; chips: {(r, c): (fill, ink)}; colors: {(r, c): ink}."""
        t = self.d.add_table(rows=1, cols=len(headers))
        self._table_style(t, pad=pad)
        for i, hd in enumerate(headers):
            c = t.rows[0].cells[i]
            self._shade(c, C_NAVY)
            par = c.paragraphs[0]
            if i in align_right:
                par.alignment = 2
            self._run(par, hd, size=font - 0.4, bold=True, color="FFFFFF")
        for k, row in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                par = cells[i].paragraphs[0]
                if i in align_right:
                    par.alignment = 2
                fill_ink = (chips or {}).get((k, i))
                if fill_ink:
                    self._shade(cells[i], fill_ink[0])
                    self._run(par, str(v), size=font, bold=True, color=fill_ink[1])
                else:
                    if k % 2 == 1:
                        self._shade(cells[i], "F7F7F5")
                    self._run(par, str(v), size=font, color=(colors or {}).get((k, i)),
                              bold=bold_col0 and i == 0)
        self._widths(t, widths)
        return t

    def card(self, title, tag, lines, accent, gap=True):
        """A stock card: title row and labelled lines, with a coloured left edge."""
        t = self.d.add_table(rows=1, cols=1)
        self._table_style(t, inside_h=False, outer=False, pad=(35, 110))
        c = t.cell(0, 0)
        trPr = t.rows[0]._tr.get_or_add_trPr()
        trPr.append(self.El("w:cantSplit"))
        self._shade(c, "F7F8FA")
        self._cell_border(c, left=(24, accent))
        par = c.paragraphs[0]
        self._run(par, title, size=10.5, bold=True, color=C_NAVY)
        self._run(par, "   " + tag, size=8.6, color=C_INK2)
        for label, text in lines:
            q = c.add_paragraph()
            q.paragraph_format.space_after = self.Pt(0.5)
            q.paragraph_format.line_spacing = 1.0
            self._run(q, label + "  ", size=8.3, bold=True, color=C_NAVY)
            self._run(q, text, size=8.3)
        self._widths(t, [6.9])
        if gap:
            self.spacer(4)

    def img(self, path, width=6.9):
        if path and os.path.exists(path):
            self.d.add_picture(path, width=self.Inches(width))
            self.d.paragraphs[-1].paragraph_format.space_after = self.Pt(2)

    def small(self, text, size=7.6, color=C_INK2, italic=False):
        return self.p(text, size=size, color=color, italic=italic, after=2)


# ═══════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════

def _period_label(D):
    s = pd.Timestamp(D["period"]["first"])
    return f"{s:%d %b} to {pd.Timestamp(D['date']):%d %b %Y}"


def _period_short(D):
    s, e = pd.Timestamp(D["period"]["first"]), pd.Timestamp(D["period"]["end"])
    return f"{s:%d}–{e:%d %b}" if s.month == e.month else f"{s:%d %b}–{e:%d %b}"


def pick_cards(D, k=3):
    """Cards show where today's news adds something: flagged stocks (⚑) first,
    then the strongest positive or negative readings; the biggest movers fill
    any remaining place (and all three on a day without a reading)."""
    rows = list(D["rows"].values())
    order = {"ELEVATED": 0, "NOTABLE": 1}
    strength = lambda r: -abs(float(r.get("net") or 0))
    picks = sorted([r for r in rows if r["attention"] in order], key=lambda r: (order[r["attention"]], strength(r)))

    def add(r):
        if r is not None and r not in picks and len(picks) < k:
            picks.append(r)
    for r in sorted([r for r in rows if r["sentiment"] != "neutral"], key=strength):
        add(r)
    by_week = sorted(rows, key=lambda r: -abs(r["period_ret"] if not np.isnan(r["period_ret"]) else 0))
    add(by_week[0])
    add(max(rows, key=lambda r: abs(r["move"])))
    add(by_week[1])
    return picks[:k]


def flag_reason(r):
    """Why a stock carries ⚑ (ELEVATED attention), in words."""
    return {"lag": "the price went against the news and the model sides with the news",
            "contradicted": "the price went against the news and the model sides with the price",
            "model_only": "there is no news signal, but the model holds a firm view"}.get(
        r["archetype"], "the pattern engine rates it worth a closer look")


def cards_from_news(D):
    return [r for r in pick_cards(D) if r["sentiment"] != "neutral" or r["attention"] in ATTENTION_FLAG]


MIN_RECORD = 20     # matured live forecasts needed before a record is quoted


def pct0(x):
    return "—" if x is None or x != x else f"{x * 100:.0f}%"


def _in_tens(rate):
    return f"about {int(round(rate * 10))} in 10"


def _calm(D):
    return [o for o in sorted(D["outlook"].values(), key=lambda o: o["move_rank"])
            if o["activity"] == "quieter than most"]


def _calm_basis(D):
    calm = _calm(D)
    return calm[0].get("basis", "model") if calm else "model"


def _rank_key(r):
    net = r.get("net")
    if net is None:
        net = {"bullish": 0.35, "bearish": -0.35}.get(r["sentiment"], 0.0)
    return (-float(net), -news_count(r))


def _names(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _sign_col(v):
    return "1C5CAB" if v > 0.3 else "A12D2C" if v < -0.3 else C_INK2


def page_glance(doc, D):
    from . import narrative as N
    rows = list(D["rows"].values())
    n = D["period"]["sessions"]
    avg_today = float(np.nanmean([r["move"] for r in rows]))
    up = sum(r["move"] >= 0.05 for r in rows)
    dn = sum(r["move"] <= -0.05 for r in rows)
    big = max(rows, key=lambda r: abs(r["period_ret"]))
    pos = [short(r["name"]) for r in sorted(rows, key=_rank_key) if r["sentiment"] == "bullish"]
    ev = D["evidence"]

    doc.header_band("SGX Daily Signal Report",
                    f"{pd.Timestamp(D['date']):%A %d %B %Y}  ·  data to the close",
                    ["15 SGX stocks  ·  7 sectors", f"Last {n} trading days: {_period_short(D)}",
                     f"Sections: 1 today · 2 news & last {n} days · 3 next 3 days · 4 system · 5 findings"])
    if D["mode"] == "sample":
        if D.get("source") == "snapshot" and D.get("calm_source") == "stored":
            doc.note(f"SAMPLE rebuilt offline from the saved copy of the project database for "
                     f"{pd.Timestamp(D['date']):%d %b %Y}: every figure, forecast and chance is as stored that day.")
        else:
            doc.note(f"SAMPLE built from stored project records. The model used for 'calmer than most' was retrained "
                     f"on {pd.Timestamp(D['model']['trained_at']):%d %b %Y} with data to "
                     f"{pd.Timestamp(str(D['model']['train_window']).split('..')[-1]):%d %b %Y}, so this date is "
                     "inside its training period for that column. The live report also prints each stock's AI news "
                     "summary and headlines.")

    doc.h("1 · Today in 30 seconds")
    nar = D.get("narrative") or {}
    lines = N.market_read(D)
    refs = N.MARKET_READ_REFS
    if nar.get("summary") and len(nar["summary"]) == len(lines):
        # each AI line already names its topic, so no bold lead (it would repeat it); pages added by code
        for text, ref in zip(nar["summary"], refs):
            doc.bullet(text.rstrip() + ref)
        kept = [a.strip() == t.strip() for a, (_, t) in zip(nar["summary"], lines)]
        doc.small((f"Drafted by code from Tasks 1-6; the AI writer ({nar.get('model')}, the same model as Task 5) "
                   "checked it and kept the wording" + ("" if all(kept) else f", rewording {len(kept) - sum(kept)} "
                   f"of {len(kept)} lines") + "; every number was filled in by code and the text passed the "
                   "report's checks."), size=7.2)
    elif nar.get("summary"):
        doc.p(" ".join(nar["summary"]))
        doc.small(f"Written by AI ({nar.get('model')}, the same model as Task 5) from a code-written draft of these "
                  "facts; every number was filled in by code and the text passed the report's checks.", size=7.2)
    else:
        for (lead, text), ref in zip(lines, refs):
            doc.bullet(text.rstrip() + ref, lead=lead)

    doc.small("Which parts can you rely on? Each kind of statement was tested on past data (results in section 5).",
              size=8, color=C_NAVY)
    doc.chips_row([("good", "Prices and changes: fact (sections 1-2)"),
                   ("info", "News & signals: describe today, not a forecast (sections 1-2)"),
                   ("good", f"Normal 3-day swing: reliable, held {ev['range']['range_coverage'] * 100:.0f} "
                            "times in 100 (section 3)"),
                   ("warn", "Calmer than most: promising, still being confirmed (section 3)"),
                   ("bad", "Up-or-down lean: weak; only ✓ calls carry information (section 3)")], size=7.3)

    doc.h("All 15 stocks today")
    mkt = market_signals(D)
    doc.p("Ranked by today's news-and-signals reading, most positive first. "
          + (f"Market-wide today: {'; '.join(mkt)}; this applies to all 15 and is not repeated below."
             if mkt else "No market-wide signal fired today."), size=8.6, color=C_INK2, after=2)
    table, chips, colors = [], {}, {}
    thr = D.get("t5_threshold") or 0.6
    for i, r in enumerate(sorted(rows, key=_rank_key)):
        star = " ★" if any("corporate event" in str(s.get("factor", "")).lower()
                           for s in card_signals(r.get("signals"))) else ""
        flag = " ⚑" if r["attention"] in ATTENTION_FLAG else ""
        tone, n_art = news_tone(r), news_count(r)
        table.append([f"{i + 1}", f"{short(r['name'])}{star}{flag}", pct(r["move"]), pct(r["period_ret"]),
                      EVIDENCE_WORD[r["sentiment"]], f"{tone_label(r)} ({n_art})", chart_short(r),
                      evidence_vs_price(r["sentiment"], r.get("same_day"))])
        chips[(i, 4)] = CHIP[r["sentiment"]]
        colors[(i, 2)] = _sign_col(r["move"])
        colors[(i, 3)] = _sign_col(r["period_ret"])
        if tone in ("few", "none"):
            colors[(i, 5)] = "A12D2C"
    doc.table(["#", "Stock", "Today", _period_short(D), "News & signals", "News today", "Price chart",
               "Price vs news"], table,
              widths=[0.25, 1.52, 0.46, 0.55, 0.72, 0.96, 1.3, 1.14], font=7.6, chips=chips, colors=colors,
              align_right=(2, 3), pad=(22, 60))
    doc.small("The ranking describes today: in our tests it did not predict which stocks did better the next day, "
              "so it is not a list of stocks to pick. News today: the tone and number of company articles collected "
              "today. Price chart: chart signals that fired. Price vs news: whether today's price moved with the "
              f"reading (a move within ±{thr:.1f}%, a normal day, counts as no reaction). ★ more company-event news "
              f"than usual (earnings, dividends, upgrades). ⚑ worth a closer look "
              f"({sum(r['attention'] in ATTENTION_FLAG for r in rows) or 'none'} today; meaning in the glossary).",
              size=7.4)


def page_week(doc, D, chart):
    n = D["period"]["sessions"]
    doc.h(f"2 · The last {n} trading days and what may explain them", new_page=True,
          sub="Each square is one day's news-and-signals reading for a stock; the bar is its price change over "
              f"{_period_short(D)}. Stocks are grouped by sector, best sector first.")
    doc.img(chart, width=6.9)
    cards = pick_cards(D)
    news = cards_from_news(D)
    if news:
        doc.h2("Where today's news adds something")
        doc.small(("The stocks with a positive or negative news-and-signals reading today (⚑ first, strongest "
                   "first)" + (", then the biggest movers." if len(news) < len(cards) else ".")), size=7.6)
    else:
        doc.h2("The biggest movers")
        doc.small("No stock had a positive or negative news-and-signals reading today, so the cards show the "
                  "biggest movers.", size=7.6)
    if any(re.search(r"geopolitic|political", r.get("summary") or "", re.I) for r in cards):
        doc.small("Task 5's AI summaries mention a 'geopolitical backdrop' and 'political news volume': those two "
                  "rules fire every day and cancel out, so they carry no information (section 5).", size=7.4)
    for ci, r in enumerate(cards):
        lines = []
        short_ai = ((D.get("narrative") or {}).get("cards") or {}).get(r["ticker"])
        if short_ai:
            lines.append(("In short (AI-written, checked)", short_ai))
        lines += [("What happened", card_happened(r, D)),
                  ("Today: news vs price", today_text(r, D)),
                  (f"Last {n} days in context", context_text(r, D)),
                  ("News today", news_text(r))]
        heads = headline_text(r)
        if heads:
            lines.append(("Headlines", heads))
        lines += [("Price chart", chart_text(r)), ("Task 5 reading", reading_text(r, D)),
                  ("Task 5 rules", rules_text(r))]
        summ = clean_t5_summary(r.get("summary"))
        if summ and summary_is_clean(summ):
            lines.append(("AI summary (Task 5: rules, two ML models, contradiction check; checked)", summ))
        lines.append(("Task 6 lean", card_forecast(r, D)))
        accent = C_BLUE if r["period_ret"] > 0 else "E34948"
        doc.card(f"{r['name']}", f"{r['sector']}  ·  {pct(r['period_ret'])} over {_period_short(D)}", lines,
                 accent, gap=ci < len(cards) - 1)


def page_next(doc, D):
    f, q, ev = D["findings"], D["evidence"]["quiet"], D["evidence"]
    doc.h("3 · The next 3 trading days: how much each stock may move", new_page=True,
          sub="The normal swing and the calmer stocks have held up in testing. The up-or-down lean is weak on its "
              "own; ✓ marks the calls whose record makes them worth a look.")
    calm = _calm(D)
    calm_set = {o["ticker"] for o in calm}
    looks = sorted(D["outlook"].values(), key=lambda o: o["range"]["half_width_pct"])
    rows, chips, colors = [], {}, {}
    fc_col = {"bullish": "1C5CAB", "bearish": "A12D2C", "neutral": C_INK2}
    for i, o in enumerate(looks):
        r = D["rows"][o["ticker"]]
        hw = o["range"]["half_width_pct"]
        p = r.get("t6_prob")
        rows.append([short(o["name"]), f"±{hw:.1f}%", "✓ calmer" if o["ticker"] in calm_set else "",
                     FORECAST_WORD[r["t6_class"]], "—" if p is None else (">99%" if p > 0.995 else f"{p * 100:.0f}%"),
                     record_cell(r)])
        colors[(i, 3)] = fc_col.get(r["t6_class"])
        chips[(i, 5)] = (VERDICT["good"][1], VERDICT["good"][2]) if not r["withheld"] else (VERDICT["bad"][1],
                                                                                           VERDICT["bad"][2])
        if o["ticker"] in calm_set:
            chips[(i, 2)] = ("DCEAFB", "1C5CAB")
    t6t = doc.table(["Stock", "Normal 3-day swing", "Calmer than most", "Task 6 lean", "Chance",
                     "Record vs simple rule"], rows,
                    widths=[1.85, 0.95, 1.0, 1.0, 0.7, 1.4], font=8.0, chips=chips, colors=colors,
                    align_right=(1, 4), pad=(26, 60))
    thr = D.get("t6_threshold")
    for row in t6t.rows[:-1]:          # heading and table move to the next page together if they do not fit
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.keep_with_next = True
    doc.small("Normal 3-day swing: about 2 times in 3 the 3-day change stays inside it (last 20 trading days). "
              "Calmer than most: the five highest chances of no big move. Task 6 lean: the model's most likely "
              f"3-day outcome (rise above +{thr:.2f}%, fall below −{thr:.2f}%, otherwise no big move) and the "
              "probability it gave. Record vs simple rule: percentage points more (or fewer) of its calls on this "
              "stock that were right over the last 30 days than always guessing the stock's most common past "
              "outcome; ✓ no more than 5 points behind, ✗ further behind. Sorted calmest first.", size=7.4)

    doc.h2("What these numbers are worth")
    doc.bullet(f"Held {ev['range']['range_coverage'] * 100:.0f}% of the time in {ev['range']['n']} tests (about 68% "
               "expected). Use it as a yardstick: a 3-day move bigger than a stock's swing is unusual for that stock "
               "and worth checking the news for; it is not a signal by itself.", lead="Normal 3-day swing. ", size=8.6)
    if calm:
        names = _names([short(o["name"]) for o in calm])
        doc.bullet(f"{names}: the five with the highest chance of no big move. In tests over {q['dates']} trading "
                   f"days, the five picked this way moved more than 1% {_in_tens(q['quiet_move_rate'])} times, the "
                   f"other ten {_in_tens(q['rest_move_rate'])} times. If you prefer less day-to-day movement, these "
                   "have moved least often.", lead="Calmer than most. ", size=8.6)
    chances = [r["t6_prob"] for r in D["rows"].values() if r.get("t6_prob") is not None]
    span = (f" Today's chances run from {min(chances) * 100:.0f}% to {max(chances) * 100:.0f}%, so no stock stands out."
            if chances and max(chances) < 0.6 else "")
    if f["n"] >= MIN_RECORD:
        ge90 = (f" Before its probabilities were refitted, chances of 90% or more were right only "
                f"{pct0(f['ge90_acc'])} of the time." if f["ge90_n"] else "")
        doc.bullet(f"Weak on its own: right {pct0(f['acc'])} of {f['n']} past calls, against "
                   f"{pct0(f['neutral'])} for always guessing 'no big move'. The ✓ mark separates the better calls "
                   f"(right {pct0(f['shown'])}) from the rest ({pct0(f['hidden'])}), so only ✓ calls carry "
                   f"information.{ge90}{span}", lead="Up-or-down lean. ", size=8.6)
    else:
        doc.bullet(f"Only {f['n']} live forecasts had matured by this date, too few to quote a record; treat every "
                   f"call as unproven.{span}", lead="Up-or-down lean. ", size=8.6)


def page_system(doc, D):
    rows = list(D["rows"].values())
    it, f = D["inputs_today"], D["findings"]
    doc.h("4 · The system behind this report", new_page=True,
          sub="Seven tasks run in sequence every trading day. Every number on the pages above comes from one of "
              "them; nothing is written by hand.")
    doc.table(["Stage", "What it does for this report"], [
        ["Tasks 1–4 — collect", f"Today's raw facts. {it['articles']} company news articles (each read as "
         f"positive, negative or neutral by FinBERT, a finance language model) and {it['political']} political "
         f"articles; price-chart indicators (RSI, MACD, Bollinger bands, VWAP, weekly trend); the market "
         f"backdrop (VIX {it['vix']:.1f}, US yield spread); Singapore ETF options positioning (put/call "
         f"{it['pc_ratio']:.2f})."],
        ["Task 5 — read the evidence", "What the news and indicators say on top of the price itself: one reading "
         "per stock (sections 1–2), built in four layers. (1) Weighted rules score the day's evidence — the scores "
         "and rules on the cards. (2) Two independent machine-learning models re-read the same inputs as a "
         "second opinion. (3) A contradiction check compares all layers and flags internally mixed readings "
         "(named on the cards when strongly mixed). (4) An AI writer (an open-source model served by Groq) drafts the news "
         "summary; it may only phrase what the rules computed, and a summary that fails the checks is replaced "
         "by a fixed template."],
        ["Task 6 — forecast", "The 3-day forecast and chance for every stock (section 3). Three model families "
         "(logistic regression, random forest, XGBoost) are re-evaluated every week in a champion–challenger "
         f"setup. A replay of that weekly choice under {f['sw_n']} settings moved accuracy only between "
         f"{f['sw_lo'] * 100:.1f}% and {f['sw_hi'] * 100:.1f}% ({f['sw_sw_lo']}–{f['sw_sw_hi']} switches), and "
         f"the pooled winner led per stock on just {f['r13_leads']} of {f['r13_n']}: switching added churn, not "
         "accuracy. The serving model is therefore pinned (random forest) and the weekly evaluation is kept as "
         "monitoring. Each forecast carries its own per-stock record — the ✓ / ✗ column."],
        ["Task 7 — check and write", "Sorts each stock into one of 11 patterns (below; shown on the cards and "
         "table in plain words as 'price vs news'), flags what deserves a closer look (⚑), picks the cards, and "
         "draws the day's market read: where the moves were, whether company news, the sector or the whole market "
         "moved with them, and what the backdrop and Task 6 add. Every kind of statement carries its tested record "
         "(section 5). An AI writer, the same model as Task 5, rewrites the code-written market read in plainer words "
         "and drafts each card's 'In short' line. It may not write a single digit (code fills in every number), "
         "and a draft that breaks any rule — no advice, no targets, no claimed certainty, no number or stock it "
         "was not given, no rise word next to a fall, no claim that the news explains a move the report says it "
         "does not, no more than little weight on Task 6 — is replaced by the template text."],
    ], widths=[1.15, 5.75], font=8.2, bold_col0=True, pad=(35, 70))
    nar = D.get("narrative") or {}
    doc.small(("AI-written text in this report: the market read and 'In short' lines (Task 7) and the news "
               "summaries (Task 5), all checked; every number, table and verdict is computed by code."
               if nar.get("mode") == "llm" else
               "This copy uses the template text throughout (no AI call was made). With an API key, the same model "
               "as Task 5 rewrites the market read and writes the 'In short' lines; numbers and tables are unchanged."),
              size=7.6)

    doc.h2("Today's patterns: how news, price and forecast line up")
    pat = {}
    for r in rows:
        pat.setdefault(r["archetype"], []).append(short(r["name"]))
    doc.table(["Pattern", "Stocks today", "Meaning"], [
        [f"{PATTERN_NAME.get(a, a)} ({len(names)})", ", ".join(names), PATTERN_WORD.get(a, "")]
        for a, names in sorted(pat.items(), key=lambda kv: -len(kv[1]))],
        widths=[1.25, 2.3, 3.35], font=7.8, bold_col0=True, pad=(26, 60))
    warn = sum(1 for r in rows if r.get("warning") == "high")
    mild = sum(1 for r in rows if r.get("warning") == "mild")
    extra = (f" Task 5's contradiction check rated {warn} of 15 readings strongly mixed (named on the cards) and "
             f"{mild} mildly mixed today.")
    doc.small("The other patterns in the 11-pattern engine did not occur today; the full set is defined in the "
              f"final report, Appendix B.{extra}", size=7.6)


def page_findings(doc, D):
    f, ev, q = D["findings"], D["evidence"], D["evidence"]["quiet"]
    doc.h("5 · What three months of testing show", new_page=True,
          sub="Each part of the system was tested on the project's stored data before it was allowed into this "
              "report. Results across all 15 stocks; forecast records count only outcomes known by the report date, "
              "research results are from the whole project:")
    rows = [
        ["Can today's news and signals pick stocks?",
         f"The ranking did not predict next-day returns against the other stocks (rank correlation "
         f"{f['r4_ic']:+.2f} over {f['r4_days']} days).", "info", "Describes only",
         "Use it to understand the day, not to pick stocks"],
        ["Was the news reading balanced?",
         f"An audit found one rule (company-event news) firing on {f['event_fired'] * 100:.1f}% of readings, so "
         f"readings leaned positive. Fixed on 21 Sep: positive readings {f['pos_v7'] * 100:.0f}% → "
         f"{f['pos_v8'] * 100:.0f}%, with no loss of information.", "good", "Found and fixed",
         "Readings are more balanced; negative ones stay rare"],
        ["Can the model forecast up or down?",
         (f"Live calls right {pct0(f['acc'])} of {f['n']} vs {pct0(f['neutral'])} for always 'no big move'. "
          if f["n"] >= MIN_RECORD else f"Only {f['n']} live calls had matured by this date. ")
         + f"With honest probabilities, RandomForest, XGBoost and logistic regression all scored "
         f"{f['oof_lo'] * 100:.1f}–{f['oof_hi'] * 100:.1f}% vs {f['oof_neutral'] * 100:.1f}%.", "bad",
         "Not reliable", "Three-day direction is not predictable from these inputs"],
        ["Were its stated chances honest?",
         (f"When the live model said 90% or more it was right {pct0(f['ge90_acc'])} of the time "
          f"({f['ge90_n']} calls): its calibration had been fitted on its own training rows. "
          if f["ge90_n"] >= MIN_RECORD else "Its calibration had been fitted on its own training rows. ")
         + "Refitted out-of-fold, no model says 90%.", "warn", "Found and fixed", "Never read a high chance as near-certain"],
        ["Does marking weak forecasts help?",
         (f"Calls on stocks where the model matched a simple rule (✓) were right {pct0(f['shown'])}; the "
          f"others {pct0(f['hidden'])}." if f["n"] >= MIN_RECORD else
          f"Only {f['n']} live calls had matured by this date, too few to compare."), "good", "Works as a filter", "Look at ✓ calls only, and still with care"],
        ["Does switching models weekly help?",
         f"Replaying the weekly champion choice under {f['sw_n']} settings gave {f['sw_lo'] * 100:.1f}–"
         f"{f['sw_hi'] * 100:.1f}%: switching added churn, not accuracy.", "info", "No gain",
         "One model (RandomForest) is kept fixed"],
        ["How much will a stock move?",
         f"The normal 3-day swing held {ev['range']['range_coverage'] * 100:.0f}% of the time "
         f"({ev['range']['n']} tests; about 68% expected).", "good", "Reliable",
         "Judge whether a move is unusual for that stock"],
        ["Which stocks will be calmer?",
         f"The five with the highest chance of no big move moved over 1% {_in_tens(q['quiet_move_rate'])} times, the others "
         f"{_in_tens(q['rest_move_rate'])} ({q['dates']} test days). The rule was set after looking at the data.",
         "warn", "Promising", "A hint where less is likely to happen"],
        ["What do seven years of prices say?",
         f"A stock that beat the other 14 on a day tended to lag slightly the next day, 2019-2025 (rank "
         f"correlation {f['rev_ic']:+.3f}, t {f['rev_t']:.1f}), but by less than trading costs.", "info",
         "Real but small", "Do not chase a one-day jump"],
    ]
    table, chips = [], {}
    for k, (what, found, key, verdict, use) in enumerate(rows):
        icon, fill, ink = VERDICT[key]
        table.append([what, found, f"{icon}  {verdict}", use])
        chips[(k, 2)] = (fill, ink)
    doc.table(["What was tested", "What we found", "Verdict", "What it means for you"], table,
              widths=[1.3, 3.15, 0.95, 1.5], font=7.5, chips=chips, bold_col0=True, pad=(22, 60))

    doc.h2("Limitations")
    for t in [
        "About three months of data in one market phase: every rate above could change as more data arrives.",
        "News readings rarely turn negative: bearish rules fire less often and weigh less. No negative reading "
        "means no negative signal was found, not that there was no bad news.",
        "Free data: options data is one Singapore stock-market fund shared by all 15 stocks, and news publication "
        "times have not been checked against the report's cut-off.",
        "One political-news rule counts every day as positive (it reads the news classifier's confidence, not "
        "whether the news was good or bad), and its counterweight fires every day too. The two are summed and "
        "marked 'unreliable'; Task 5's AI summaries still mention them.",
        "The news search often returns older articles, so only headlines from the last 30 days are shown; some "
        "stocks show none on a given day.",
    ]:
        doc.bullet(t, size=7.8)

    doc.h2("Words used in this report")
    doc.table(["Word", "Meaning"], GLOSSARY, widths=[1.5, 5.4], font=7.2, bold_col0=True, pad=(18, 60))
    doc.small("General information about 15 listed companies, generated automatically from stored data. It is not "
              "financial advice or a recommendation to deal in any security, and it takes no account of your "
              "circumstances. Past patterns do not guarantee future results.", size=7.4, italic=True)


def reviewer_notes(D):
    """How the report was produced, for reviewers: saved next to the report
    (T7_reviewer_notes_<date>.md), not printed in it."""
    rows = list(D["rows"].values())
    it = D["inputs_today"]
    nar = D.get("narrative") or {}
    out = [f"# T7 Daily Signal Report {D['date']}: how it was produced", "",
           f"Generated {datetime.now(SGT):%Y-%m-%d %H:%M} SGT; mode {D['mode']}.", "", "## Steps", ""]
    out += [f"- Task 1 news: {it['articles']} company articles and {it['political']} political articles, FinBERT-scored; event tags.",
            "- Task 2 technical: RSI, MACD, Bollinger bands (20, 2σ), VWAP and weekly trend per stock.",
            f"- Task 3 macro: regime code {it['regime']}; VIX {it['vix']:.1f}; US yield spread.",
            f"- Task 4 options: Singapore ETF (EWS) put/call volume ratio {it['pc_ratio']:.2f}"
            f"{', positioning negative' if it['options_bearish'] else ''}.",
            f"- Task 5 evidence: rule layer {D['versions']['rule_layer']}: "
            f"{sum(r['sentiment'] == 'bullish' for r in rows)} positive, {sum(r['sentiment'] == 'bearish' for r in rows)} "
            f"negative, {sum(r['sentiment'] == 'neutral' for r in rows)} neutral.",
            "- Task 6 forecast: the stored live champion forecast and probabilities for each stock, with its record "
            "against a time-ordered simple rule (✓ / ✗); 'calmer than most' from "
            + ("the same stored probabilities." if D.get("calm_source") == "stored" else
               f"the bundled {D['model']['name']} model (trained {D['model']['trained_at']})."),
            f"- Task 7: patterns, attention, cards, market read, compliance checks R1-R7; narrative "
            + (f"AI ({nar.get('model')}) with checks N1-N11." if nar.get("mode") == "llm" else "template."), "",
            "## Provenance", ""]
    out += [f"- {p}" for p in D.get("provenance") or []]
    days = D["period"].get("days") or []
    rd = D["period"].get("reading_dates") or []
    missing = [d for d in days if d not in rd]
    out.append(f"- Period: close of {pd.Timestamp(D['period']['start']):%d %b} to close of {pd.Timestamp(D['date']):%d %b %Y}"
               + (f"; no evidence reading stored for {', '.join(f'{pd.Timestamp(x):%a %d %b}' for x in missing)}" if missing else ""))
    if nar.get("retried"):
        out.append("- AI blocks that passed only on a second attempt: " + ", ".join(nar["retried"]))
    if nar.get("flags"):
        out.append("- AI blocks replaced by the template: " + "; ".join(nar["flags"]))
    pat = pd.Series([PATTERN_NAME.get(r["archetype"], r["archetype"]) for r in rows]).value_counts()
    att = pd.Series([r["attention"].title() for r in rows]).value_counts()
    gaps = [r["edge"] for r in rows if r["withheld"] and r["edge"] is not None]
    ev = D["evidence"]
    out += ["", "## Measures", "",
            f"- Patterns today: {', '.join(f'{k} {v}' for k, v in pat.items())}",
            f"- Attention today: {', '.join(f'{k} {v}' for k, v in att.items())} (⚑ = Elevated or Notable)",
            f"- Forecasts marked ✗: {len(gaps)} of 15"
            + (f"; median record gap {abs(np.median(gaps)) * 100:.0f} points below the simple rule" if gaps else ""),
            f"- Outlook record (R14, {'as of ' + ev['as_of'] if ev.get('as_of') else 'frozen'}): {ev['n_dates']} dates; "
            f"calmer five moved >1% {ev['quiet']['quiet_move_rate'] * 100:.0f}% vs {ev['quiet']['rest_move_rate'] * 100:.0f}%; "
            f"normal swing held {ev['range']['range_coverage'] * 100:.0f}%",
            "", "## Terms", "",
            "- News & signals: Task 5 net rule score beyond ±0.35. News today: (positive − negative) / articles beyond ±0.15; "
            "'too few' below 3 articles.",
            f"- Price vs news: today's move against Task 5's label threshold ±{D.get('t5_threshold') or 0.6:.2f}%.",
            "- Normal 3-day swing: ±(20-day s.d. × √3). Calmer than most: five lowest 1 − P(no significant move).",
            "- ✗: the record is more than five points below a rule predicting the stock's most common matured outcome."]
    return "\n".join(out) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════════════════

def render(D, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart = chart_week(D, str(out_dir / f"week_{D['date']}.png"))
    doc = Doc(footer=f"SGX Daily Signal Report  ·  {pd.Timestamp(D['date']):%d %b %Y}  ·  "
                     "general information, not financial advice")
    page_glance(doc, D)
    page_week(doc, D, chart)
    page_next(doc, D)
    page_system(doc, D)
    page_findings(doc, D)
    tag = "SAMPLE_" if D["mode"] == "sample" else ""
    path = out_dir / f"T7_Daily_Signal_Report_{tag}{D['date']}.docx"
    doc.d.save(path)
    (out_dir / f"T7_reviewer_notes_{tag}{D['date']}.md").write_text(reviewer_notes(D))
    keep = ("ticker", "name", "sector", "move", "period_ret", "sentiment", "week_pos", "week_neg", "week_n",
            "t6_class", "t6_prob", "t6_source", "same_day", "archetype", "attention", "withheld", "edge", "n_prior",
            "net", "signals", "warning")
    ev = D["evidence"]
    rec = {"date": D["date"], "mode": D["mode"], "period": D["period"], "record": D["record"],
           "findings": D.get("findings"), "narrative": D.get("narrative"),
           "outlook_record": {"window": ev.get("window"), "n_dates": ev.get("n_dates"), "as_of": ev.get("as_of"),
                              "quiet_move_rate": ev["quiet"]["quiet_move_rate"],
                              "rest_move_rate": ev["quiet"]["rest_move_rate"],
                              "range_coverage": ev["range"]["range_coverage"]},
           "rows": {t: {k: r.get(k) for k in keep} for t, r in D["rows"].items()},
           "cards": [r["ticker"] for r in pick_cards(D)], "calm_source": D.get("calm_source"),
           "outlook": {t: {k: o.get(k) for k in ("move_rank", "activity", "range", "vol_quiet",
                                                  "move_by_group", "top_group", "basis")}
                       for t, o in D["outlook"].items()}}
    (out_dir / f"T7_report_input_{tag}{D['date']}.json").write_text(json.dumps(rec, indent=1, default=float))
    return path


def build_sample(root=None, out_dir=None, period_sessions=PERIOD_SESSIONS, date=SAMPLE_DATE,
                 api_key=None, llm_settings=None):
    from .configuration import ROOT
    from . import narrative as N
    root = Path(root or ROOT)
    D = gather_sample(root, date=date or SAMPLE_DATE, period_sessions=period_sessions)
    saved = D.pop("narrative_saved", None)
    if saved and not N.matches_layout(saved, D):
        print("  Saved AI text was written for an earlier layout of the report; using the template text.")
        saved = None
    D["narrative"] = (N.build(D, api_key, llm_settings) if api_key or not saved else saved)
    nar = D["narrative"]
    print(f"  AI narrative: {'written by ' + str(nar['model']) if nar['mode'] == 'llm' else 'off (template text)'}"
          + (f"; {len(nar['flags'])} block(s) replaced: {'; '.join(nar['flags'][:4])}" if nar["flags"] else ""))
    return render(D, out_dir or root / "outputs" / "sample_report")


def build_live(ns, cfg, report_date=None, save_snapshot=False, api_key=None):
    from .configuration import ROOT
    from . import narrative as N
    D = gather_live(ns, cfg, report_date, save=save_snapshot)
    D["narrative"] = N.build(D, api_key, cfg.get("llm"))
    if save_snapshot:
        path = snapshot_path(ROOT, D["date"])
        blob = json.loads(path.read_text())
        blob["narrative"] = D["narrative"]
        path.write_text(json.dumps(blob, default=_jsonable, indent=1))
    nar = D["narrative"]
    print(f"  AI narrative: {'written by ' + str(nar['model']) if nar['mode'] == 'llm' else 'off (template text)'}"
          + (f"; {len(nar['flags'])} block(s) replaced: {'; '.join(nar['flags'][:4])}" if nar["flags"] else ""))
    return render(D, ROOT / "outputs" / "report")
