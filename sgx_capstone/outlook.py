"""Next-three-session outlook: what the system can say about each stock for the
coming three trading sessions, why the Task 6 model says it, and how often
statements of that kind have held up.

The outlook makes three statements, in decreasing order of reliability.

1. Typical range. +/- s * sqrt(3), where s is the standard deviation of the
   stock's last 20 daily returns. Record (R14): the realised three-session
   change fell inside this range on 73.8% of 420 stock-days (about 68%
   expected for a normal distribution).
2. Activity. Each day the five stocks with the highest model chance of no
   significant move (lowest move score 1 - P(neutral)) are called "quieter
   than most". Record (R14, out-of-fold, 31 dates, event = three-session
   move above 1%): those five moved in 41.9% of cases against 56.5% for the
   other ten, with the lower rate on 22 of 31 dates (2 tied). A ranking from
   recent volatility alone does about as well; the report says so. The
   model ranking is used only when the serving model has that record;
   otherwise activity is read from volatility alone.
3. Direction. Not stated. Among stocks that did move, the model's up-versus-
   down lean separated rises from falls no better than chance (AUC 0.48).

The explanation of statement 2 is exact for the model: group Shapley values
over five input groups, measured against an "average day" (the training
means held in the artifact's scaler). The five contributions add up to the
difference between today's estimate and the average-day estimate. The
sentence is written by template. A language-model version with checks O1-O7
(validate_explanation) was built and tested; the checks confirm wording and
that each number exists in the facts, not that a number is attached to the
right thing, so numbers and effect directions are not left to the model.
validate_explanation is kept, with a test that documents that limit.

Nothing in this module reads a database or the network.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 3
VOL_WINDOW = 20
VOL_MIN = 5    # same minimum as research R14
N_QUIET = 5

# Five input groups a reader can recognise. Every T6 feature belongs to one.
FEATURE_GROUPS = {
    "company news": ["news_bullish_ratio", "news_bearish_ratio", "avg_sentiment",
                     "n_articles", "n_bullish_events", "n_bearish_events"],
    "political news": ["n_political", "political_direction"],
    "price indicators": ["rsi", "macd_bullish_crossover", "macd_bearish_crossover",
                         "bb_above_upper", "bb_below_lower", "vwap_above",
                         "weekly_bullish"],
    "market conditions": ["regime_encoded", "vix", "yield_spread"],
    "options market": ["options_bullish", "options_bearish", "pc_vol_ratio"],
}

# Plain-language description of each feature.
# kind "level"/"count"/"share": phrase uses {today} and {avg}.
# kind "flag": (phrase when true, phrase when false); {avg} is the share of
# training days on which the flag was true.
PLAIN = {
    "news_bullish_ratio": ("share", 0, "{today} of company articles read as positive ({avg} on an average day)"),
    "news_bearish_ratio": ("share", 0, "{today} of company articles read as negative ({avg} on an average day)"),
    "avg_sentiment": ("level", 2, "news tone scores averaging {today} in confidence ({avg} on an average day)"),
    "n_articles": ("count", 0, "{today} company articles in the look-back window ({avg} on an average day)"),
    "n_bullish_events": ("count", 0, "{today} articles tagged with positive corporate events ({avg} on an average day)"),
    "n_bearish_events": ("count", 0, "{today} articles tagged with negative corporate events ({avg} on an average day)"),
    "n_political": ("count", 0, "{today} political news articles ({avg} on an average day)"),
    "political_direction": ("level", 2, "political news tone of {today} ({avg} on an average day)"),
    "rsi": ("level", 0, "an RSI momentum reading of {today}, where 30 is oversold and 70 overbought ({avg} on an average day)"),
    "macd_bullish_crossover": ("flag", 0, ("the MACD indicator had just turned upward (true on {avg} of days)",
                                            "no fresh upward MACD turn")),
    "macd_bearish_crossover": ("flag", 0, ("the MACD indicator had just turned downward (true on {avg} of days)",
                                            "no fresh downward MACD turn")),
    "bb_above_upper": ("flag", 0, ("the price above its upper Bollinger band (true on {avg} of days)",
                                    "the price inside its upper Bollinger band")),
    "bb_below_lower": ("flag", 0, ("the price below its lower Bollinger band (true on {avg} of days)",
                                    "the price inside its lower Bollinger band")),
    "vwap_above": ("flag", 0, ("the price above its volume-weighted average (true on {avg} of days)",
                                "the price below its volume-weighted average (above it on {avg} of days)")),
    "weekly_bullish": ("flag", 0, ("a weekly trend pointing upward (true on {avg} of days)",
                                    "no upward weekly trend (one is present on {avg} of days)")),
    "regime_encoded": ("level", 0, "macro regime code {today} ({avg} on an average day)"),
    "vix": ("level", 1, "the US volatility index (VIX) at {today} ({avg} on an average day)"),
    "yield_spread": ("level", 2, "a US 10-year minus 2-year yield gap of {today} points ({avg} on an average day)"),
    "options_bullish": ("flag", 0, ("options positioning leaning positive (true on {avg} of days)",
                                     "options positioning not leaning positive")),
    "options_bearish": ("flag", 0, ("options positioning leaning negative (true on {avg} of days)",
                                     "options positioning not leaning negative")),
    "pc_vol_ratio": ("level", 2, "a put/call volume ratio of {today} on the Singapore ETF ({avg} on an average day)"),
}


# ═══════════════════════════════════════════════════════════════════════════
# EVIDENCE
# ═══════════════════════════════════════════════════════════════════════════

def load_evidence(root) -> dict:
    """R14 results, written by sgx_capstone.research.r14_outlook."""
    path = Path(root) / "evidence" / "outlook_reliability.json"
    return json.loads(path.read_text())


def in_training_window(report_date: str, artifact: dict) -> bool:
    """True when the date's three-session outcome may have been in training."""
    window = str(artifact.get("train_window", ""))
    if ".." not in window:
        return False
    end = window.split("..")[1]
    return str(report_date) <= end


# ═══════════════════════════════════════════════════════════════════════════
# MODEL QUANTITIES
# ═══════════════════════════════════════════════════════════════════════════

def _classes(artifact) -> list:
    return list(artifact["label_encoder"].classes_)


def _model_input(artifact, model_name, frame):
    """LogReg needs scaled input; the tree models were trained on raw rows."""
    if model_name == "LogReg":
        return artifact["scaler"].transform(frame)
    return frame


def predict(artifact, model_name, frame: pd.DataFrame) -> pd.DataFrame:
    """Calibrated class probabilities, one row per input row."""
    est = artifact[{"LogReg": "logreg_calibrated", "RandomF": "rf_calibrated",
                    "XGBoost": "xgb_calibrated"}[model_name]]
    names = artifact["feature_names"]
    proba = est.predict_proba(_model_input(artifact, model_name, frame[names]))
    return pd.DataFrame(proba, columns=_classes(artifact), index=frame.index)


def average_day(artifact) -> pd.Series:
    """Training means of the 21 features: the reference 'average day'."""
    return pd.Series(artifact["scaler"].mean_, index=artifact["feature_names"])


def fill_missing(row: pd.Series, artifact) -> tuple[pd.Series, list]:
    """Missing inputs take the average-day value, never zero.

    Zero is not neutral for most features (an RSI of 0 is an extreme reading,
    a VIX of 0 is impossible), so a zero fill invents a signal.
    """
    base = average_day(artifact)
    row = row.reindex(base.index).astype(float)
    missing = [f for f in base.index if pd.isna(row[f])]
    row[missing] = base[missing]
    return row, missing


def group_shapley(artifact, model_name, features: pd.DataFrame) -> dict:
    """Exact Shapley values of the five feature groups, per stock.

    Players are the groups; a coalition takes today's values for its groups
    and average-day values for the rest. With five groups that is 32
    evaluations per stock, so the values are exact, not sampled. Two
    quantities are explained: the chance of a significant move (1 - P(neutral))
    and the lean (P(bullish) - P(bearish)).
    """
    names = list(FEATURE_GROUPS)
    n = len(names)
    base = average_day(artifact)
    subsets = [s for r in range(n + 1) for s in itertools.combinations(range(n), r)]
    rows, keys = [], []
    for ticker, x in features.iterrows():
        for s in subsets:
            v = base.copy()
            for gi in s:
                cols = FEATURE_GROUPS[names[gi]]
                v[cols] = x[cols].astype(float).values
            rows.append(v.values)
            keys.append((ticker, s))
    proba = predict(artifact, model_name,
                    pd.DataFrame(rows, columns=base.index))
    move = (1 - proba["neutral"]).values
    lean = (proba["bullish"] - proba["bearish"]).values
    value = {k: (m, l) for k, m, l in zip(keys, move, lean)}

    out = {}
    for ticker in features.index:
        phi_m, phi_l = {}, {}
        for i, g in enumerate(names):
            tm = tl = 0.0
            for s in subsets:
                if i in s:
                    continue
                w = (math.factorial(len(s)) * math.factorial(n - len(s) - 1)
                     / math.factorial(n))
                with_i = tuple(sorted(s + (i,)))
                tm += w * (value[(ticker, with_i)][0] - value[(ticker, s)][0])
                tl += w * (value[(ticker, with_i)][1] - value[(ticker, s)][1])
            phi_m[g], phi_l[g] = tm, tl
        full = value[(ticker, tuple(range(n)))]
        empty = value[(ticker, ())]
        out[ticker] = {"p_move": float(full[0]), "lean": float(full[1]),
                       "p_move_average_day": float(empty[0]),
                       "lean_average_day": float(empty[1]),
                       "move_by_group": phi_m, "lean_by_group": phi_l}
    return out


def feature_effects(artifact, model_name, x: pd.Series, group: str) -> list:
    """Within one group: how much the move chance changes when each single
    feature is reset to its average-day value (others kept as today).
    Used only to pick the concrete fact quoted in the explanation."""
    base = average_day(artifact)
    cols = FEATURE_GROUPS[group]
    frame = pd.DataFrame([x.values] * (len(cols) + 1), columns=base.index)
    for i, c in enumerate(cols, start=1):
        frame.loc[i, c] = base[c]
    p = 1 - predict(artifact, model_name, frame)["neutral"].values
    effects = [(c, float(p[0] - p[i])) for i, c in enumerate(cols, start=1)]
    return sorted(effects, key=lambda e: -abs(e[1]))


def describe_feature(name, value, average) -> dict:
    """A feature value next to its average-day value, as a phrase."""
    kind, dp, phrase = PLAIN[name]
    if kind == "flag":
        true = float(value) >= 0.5
        avg = f"{average * 100:.0f}%"
        text = phrase[0 if true else 1].format(avg=avg)
        today = "yes" if true else "no"
    elif kind == "share":
        today, avg = f"{value * 100:.0f}%", f"{average * 100:.0f}%"
        text = phrase.format(today=today, avg=avg)
    else:
        fmt = f"{{:.{dp}f}}"
        today, avg = fmt.format(value), fmt.format(average)
        text = phrase.format(today=today, avg=avg)
    return {"feature": name, "today": today, "average": avg, "text": text}


# ═══════════════════════════════════════════════════════════════════════════
# PRICE-BASED STATEMENTS
# ═══════════════════════════════════════════════════════════════════════════

def typical_range(daily_returns_pct: pd.Series) -> dict | None:
    """+/- s * sqrt(3) from the last 20 daily returns (percent)."""
    r = pd.Series(daily_returns_pct, dtype=float).dropna().tail(VOL_WINDOW)
    if len(r) < VOL_MIN:
        return None
    s = float(r.std(ddof=1))
    return {"daily_sd_pct": s, "half_width_pct": s * math.sqrt(HORIZON),
            "n_returns": int(len(r))}


# ═══════════════════════════════════════════════════════════════════════════
# THE OUTLOOK
# ═══════════════════════════════════════════════════════════════════════════

def build_outlook(artifact, model_name, features: pd.DataFrame,
                  daily_returns: pd.DataFrame, evidence: dict,
                  names: dict | None = None,
                  stored_probabilities: dict | None = None) -> dict:
    """One outlook per stock.

    features       DataFrame indexed by ticker, T6 feature columns (NaN allowed)
    daily_returns  DataFrame (dates x tickers) of daily % returns, ending on
                   the report date; only past returns are used
    evidence       load_evidence(root)
    stored_probabilities  {ticker: {class: p}} from task6_predictions, used
                   only to report whether the bundled model reproduces them
    """
    names = names or {}
    filled, missing = {}, {}
    for t, row in features.iterrows():
        filled[t], missing[t] = fill_missing(row, artifact)
    X = pd.DataFrame(filled).T
    shap = group_shapley(artifact, model_name, X)
    probs = predict(artifact, model_name, X)
    record = (evidence.get("by_model") or {}).get(model_name, {})
    model_has_record = bool(record.get("quiet_record"))

    rng = {t: typical_range(daily_returns[t]) if t in daily_returns else None
           for t in X.index}
    vol = pd.Series({t: (r["daily_sd_pct"] if r else np.nan) for t, r in rng.items()})
    p_move = pd.Series({t: shap[t]["p_move"] for t in X.index})
    move_rank = p_move.rank(method="first").astype(int)
    vol_rank = vol.rank(method="first")
    threshold = float(artifact["return_threshold"])

    out = {}
    for t in X.index:
        s = shap[t]
        groups = sorted(s["move_by_group"].items(), key=lambda kv: -abs(kv[1]))
        top_group = groups[0][0]
        facts = []
        for f, eff in [e for e in feature_effects(artifact, model_name, X.loc[t], top_group)
                       if abs(e[1]) >= 0.005][:2]:
            d = describe_feature(f, X.loc[t, f], average_day(artifact)[f])
            d["effect_points"] = eff * 100
            facts.append(d)
        model_quiet = int(move_rank[t]) <= N_QUIET
        vol_quiet = (not np.isnan(vol_rank.get(t, np.nan))
                     and int(vol_rank[t]) <= N_QUIET)
        if model_has_record:
            activity = "quieter than most" if model_quiet else "usual"
            basis = "model"
        else:
            activity = "quieter than most" if vol_quiet else "usual"
            basis = "volatility"
        agree = model_quiet == vol_quiet
        repro = None
        if stored_probabilities and t in stored_probabilities:
            sp = stored_probabilities[t]
            repro = float(max(abs(float(sp[c]) - float(probs.loc[t, c]))
                              for c in probs.columns if sp.get(c) is not None))
        out[t] = {
            "ticker": t, "name": names.get(t, t), "model": model_name,
            "threshold_pct": threshold,
            "probabilities": {c: float(probs.loc[t, c]) for c in probs.columns},
            "p_move": s["p_move"], "p_move_average_day": s["p_move_average_day"],
            "move_rank": int(move_rank[t]), "n_stocks": int(len(X)),
            "move_by_group": s["move_by_group"], "lean_by_group": s["lean_by_group"],
            "lean": s["lean"], "top_group": top_group, "top_facts": facts,
            "missing_features": missing[t],
            "range": rng[t], "vol_rank": None if np.isnan(vol_rank.get(t, np.nan)) else int(vol_rank[t]),
            "model_quiet": model_quiet, "vol_quiet": bool(vol_quiet),
            "activity": activity, "activity_basis": basis, "agree": agree,
            "model_has_record": model_has_record,
            "stored_max_abs_diff": repro,
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

def explanation_facts(o: dict, evidence: dict) -> dict:
    """The facts behind one explanation, rounded as displayed."""
    groups = sorted(o["move_by_group"].items(), key=lambda kv: -abs(kv[1]))
    q = evidence["quiet"]
    return {
        "stock": o["name"],
        "move_band_pct": round(o["threshold_pct"], 1),
        "chance_of_move_pct": int(round(o["p_move"] * 100)),
        "average_day_chance_pct": int(round(o["p_move_average_day"] * 100)),
        "rank_from_quietest": o["move_rank"],
        "stocks_compared": o["n_stocks"],
        "classed_as": o["activity"],
        "group_effects_points": [
            {"group": g, "points": int(round(v * 100))}
            for g, v in groups if round(v * 100) != 0][:3],
        "top_group_facts": [f["text"] for f in o["top_facts"]],
        "record": {
            "quiet_calls_moved_pct": int(round(q["quiet_move_rate"] * 100)),
            "other_stocks_moved_pct": int(round(q["rest_move_rate"] * 100)),
            "dates_tested": q["dates"],
        },
        "direction_note": "The system does not forecast direction.",
    }


def _points(n: int) -> str:
    return f"{n} percentage point" + ("" if n == 1 else "s")


def render_explanation(o: dict, evidence: dict) -> str:
    """Deterministic explanation of the activity call."""
    f = explanation_facts(o, evidence)
    effects = f["group_effects_points"]
    s1 = (f"The model puts the chance of a move larger than {f['move_band_pct']}% "
          f"in either direction over the next three sessions at "
          f"{f['chance_of_move_pct']}%, against {f['average_day_chance_pct']}% "
          f"on an average day.")
    if effects:
        lead = effects[0]
        verb = "raised" if lead["points"] > 0 else "lowered"
        s2 = f"{lead['group'].capitalize()} {verb} it most, by {_points(abs(lead['points']))}"
        if f["top_group_facts"]:
            s2 += f", with {f['top_group_facts'][0]}"
        s2 += "."
        if len(effects) > 1:
            nxt = effects[1]
            same = (nxt["points"] > 0) == (lead["points"] > 0)
            if same:
                s2 += (f" {nxt['group'].capitalize()} "
                       f"{'raised' if nxt['points'] > 0 else 'lowered'} it by a further "
                       f"{_points(abs(nxt['points']))}.")
            else:
                s2 += (f" {nxt['group'].capitalize()} worked the other way, by "
                       f"{_points(abs(nxt['points']))}.")
    else:
        s2 = "No input group moved it noticeably away from an average day."
    tail = ("so it is classed as quieter than most"
            if o["activity"] == "quieter than most" else "so it is not among the quieter stocks")
    s3 = (f"It ranks {f['rank_from_quietest']} of {f['stocks_compared']} "
          f"from the quietest today, {tail}.")
    return " ".join([s1, s2, s3])


_DIRECTION_PATTERNS = [
    r"\b(?:likely|expected|set|poised|going|bound|tipped)\s+to\s+(?:\w+\s+){0,2}?"
    r"(?:rise|fall|climb|drop|gain|decline|rally|rebound|recover|increase|decrease|go up|go down)\b",
    r"\b(?:upside|downside)\b",
    r"\b(?:bullish|bearish)\b",
]


def validate_explanation(textv: str, facts: dict) -> tuple[bool, list]:
    """Checks O1-O7 built for language-model text (kept to document their limit).

    O1-O3 reuse the report layer's advice, price-target and certainty rules.
    O4 no direction forecast. O5 every number must come from the facts.
    O6 the largest driver must be named. O7 at most 90 words.
    """
    import task7_generate as g
    flags = []
    body = textv.lower()
    for tag, pats in (("O1 advice", g._R1_PATTERNS), ("O2 target", g._R2_PATTERNS),
                      ("O3 certainty", g._R3_PATTERNS),
                      ("O4 direction", _DIRECTION_PATTERNS)):
        for pat in pats:
            if re.search(pat, body):
                flags.append(f"{tag}: /{pat}/")
    plain, pct = set(), set()
    g._collect_numbers(facts, plain, pct)
    for tok in g._NUM_RE.findall(json.dumps(facts)):
        v = round(abs(float(tok.rstrip("%"))), 2)
        plain.add(v)
        pct.add(v)
    for tok in g._NUM_RE.findall(body):
        val = round(abs(float(tok.rstrip("%"))), 2)
        pool = pct if tok.endswith("%") else plain
        if val not in pool and val not in plain:
            flags.append(f"O5 number not in facts: {tok}")
    effects = facts.get("group_effects_points") or []
    if effects and effects[0]["group"].lower() not in body:
        flags.append(f"O6 largest driver '{effects[0]['group']}' not named")
    if len(textv.split()) > 90:
        flags.append("O7 longer than 90 words")
    return not flags, flags


def explain(o: dict, evidence: dict) -> dict:
    """Return {'text', 'mode'}. The explanation is always the template."""
    return {"text": render_explanation(o, evidence), "mode": "template"}
