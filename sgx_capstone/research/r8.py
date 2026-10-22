# ═══════════════════════════════════════════════════════════════════
# R8 — T5 rl_v8 REPLAY: does an "abnormal events only" rule fix the tilt?
#
# Read-only. Run after the research cell (reuses _q, FWD, RESEARCH_DIR).
#
# Finding (R7): the bullish corporate-event rule fired on 99.7% of rl_v7
# readings and contributed +0.203 of the +0.381 average net score. Task 1
# tags events by headline keyword across ~50 articles per stock per day, so
# some headline almost always mentions "earnings" or "dividend" — the rule
# records "the stock has ongoing coverage", not "something happened today".
# It is the same failure the extractor comment describes for the old binary
# flags ("fired on ~92% of samples ... encoded 'did the ETL run today'");
# the rule layer still uses has-any-event logic.
#
# Candidate rl_v8 rule (decided BEFORE looking at the result):
#   an event rule fires only when today's number of event-tagged articles
#   for that stock exceeds its own trailing 20-trading-day mean by >= 1.
#   Same idea as [MW5] (training-mean threshold for n_political).
#   Weights, directions and every other rule unchanged.
#
# Acceptance criteria (also fixed in advance):
#   1. event rule firing rate falls below 50%
#   2. the reading distribution is no longer one-sided (some bearish readings)
#   3. next-day hit rate / rank IC not worse than rl_v7
# ═══════════════════════════════════════════════════════════════════
import json
import numpy as np
import pandas as pd

TRAIL_DAYS  = 20
EXCESS_MIN  = 1.0
THRESH      = 0.35                       # NET_SCORE_THRESHOLD (MW3), unchanged
BULL_EVT    = ("upgrade", "earnings", "dividend", "ma")
BEAR_EVT    = ("warning", "downgrade", "layoff")

# ── 1. event-article counts per stock per day, full history ──
cnt = pd.DataFrame(_q("""
    SELECT ticker, date::text,
           SUM(CASE WHEN event_flag IN ('upgrade','earnings','dividend','ma') THEN 1 ELSE 0 END),
           SUM(CASE WHEN event_flag IN ('warning','downgrade','layoff') THEN 1 ELSE 0 END),
           COUNT(*)
    FROM news_articles WHERE category = 'company'
    GROUP BY ticker, date ORDER BY ticker, date"""),
    columns=["ticker", "date", "n_bull_evt", "n_bear_evt", "n_art"])
for c in ["n_bull_evt", "n_bear_evt", "n_art"]:
    cnt[c] = cnt[c].astype(float)
for side in ["bull", "bear"]:
    col = f"n_{side}_evt"
    trail = cnt.groupby("ticker")[col].transform(
        lambda s: s.shift(1).rolling(TRAIL_DAYS, min_periods=5).mean())
    cnt[f"{side}_excess"] = cnt[col] - trail
EVT = {(r.ticker, r.date): r for r in cnt.itertuples(index=False)}
print(f"Event counts: {len(cnt)} stock-days, "
      f"mean bullish-event articles/day {cnt.n_bull_evt.mean():.1f}, "
      f"bearish {cnt.n_bear_evt.mean():.1f}, articles {cnt.n_art.mean():.1f}")

# ── 2. next-day excess returns (vs the 15-stock average) ──
_r1 = {}
for (t, d, h), r in FWD.items():
    if h == 1:
        _r1.setdefault(d, {})[t] = r
EX1 = {(t, d): r - np.mean(list(v.values()))
       for d, v in _r1.items() if len(v) >= 12 for t, r in v.items()}

# ── 3. replay every rl_v7 reading under each variant ──
def score(feats):
    b = sum(f["w"] for f in feats if f["dir"] > 0)
    s = sum(f["w"] for f in feats if f["dir"] < 0)
    net = b - s
    return net, ("bullish" if net > THRESH else "bearish" if net < -THRESH else "neutral")

rows = []
for t, d, fr in _q("""
        SELECT DISTINCT ON (ticker, analysis_date::date)
               ticker, analysis_date::date::text, features_result
        FROM llm_analyses_audit
        WHERE (features_result::jsonb)->>'_rule_layer_version' = 'rl_v7'
        ORDER BY ticker, analysis_date::date, id DESC"""):
    fr = fr if isinstance(fr, dict) else json.loads(fr)
    feats = [{"factor": f.get("factor", ""),
              "dir": 1 if f.get("direction") == "bullish" else -1,
              "w": float(f.get("baseline_weight", 0.0))} for f in fr.get("features", [])]
    is_evt = lambda f: "corporate events" in f["factor"]
    e = EVT.get((t, d))
    bull_ok = e is not None and not np.isnan(e.bull_excess) and e.bull_excess >= EXCESS_MIN
    bear_ok = e is not None and not np.isnan(e.bear_excess) and e.bear_excess >= EXCESS_MIN

    v7 = feats
    no_evt = [f for f in feats if not is_evt(f)]
    v8 = [f for f in feats if not is_evt(f)
          or (f["dir"] > 0 and bull_ok) or (f["dir"] < 0 and bear_ok)]
    rec = {"ticker": t, "date": d, "ex1": EX1.get((t, d)),
           "evt_fired_v7": any(is_evt(f) and f["dir"] > 0 for f in v7),
           "evt_fired_v8": any(is_evt(f) and f["dir"] > 0 for f in v8)}
    for name, fs in [("rl_v7", v7), ("no_events", no_evt), ("rl_v8", v8)]:
        rec[f"net_{name}"], rec[f"call_{name}"] = score(fs)
    rows.append(rec)
rp = pd.DataFrame(rows)
print(f"Replayed {len(rp)} rl_v7 readings over {rp.date.nunique()} dates "
      f"(stored-vs-replay check was OK in R4)\n")


def summarise(name):
    call, net = rp[f"call_{name}"], rp[f"net_{name}"]
    g = rp.dropna(subset=["ex1"])
    c = g[f"call_{name}"]
    dirn = g[c != "neutral"]
    hit = (np.sign(dirn["ex1"]) == np.where(dirn[f"call_{name}"] == "bullish", 1, -1)).mean() \
        if len(dirn) else np.nan
    ics = [x[f"net_{name}"].rank().corr(x["ex1"].rank())
           for _, x in g.groupby("date") if len(x) >= 10]
    return {"variant": name, "mean_net": net.mean(),
            "bull%": (call == "bullish").mean(), "neu%": (call == "neutral").mean(),
            "bear%": (call == "bearish").mean(),
            "days_with_a_bearish": rp[call == "bearish"].date.nunique(),
            "cross_sec_std": rp.groupby("date")[f"net_{name}"].std().mean(),
            "n_called": len(dirn), "hit_vs_peers_1d": hit,
            "rank_IC_1d": np.nanmean(ics)}

res = pd.DataFrame([summarise(v) for v in ["rl_v7", "no_events", "rl_v8"]])
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}")
print(res.to_string(index=False))
print(f"\n  Bullish event rule firing rate: rl_v7 {rp.evt_fired_v7.mean():.1%}  →  "
      f"rl_v8 {rp.evt_fired_v8.mean():.1%}")
ch = rp[rp.call_rl_v7 != rp.call_rl_v8]
moves = (ch.call_rl_v7 + " -> " + ch.call_rl_v8).value_counts().to_dict()
print(f"  Readings whose call changes under rl_v8: {len(ch)}/{len(rp)}  {moves}")

ok1 = rp.evt_fired_v8.mean() < 0.5
ok2 = (rp.call_rl_v8 == "bearish").any() or res.loc[2, "bull%"] < res.loc[0, "bull%"] - 0.1
ok3 = (res.loc[2, "rank_IC_1d"] >= res.loc[0, "rank_IC_1d"] - 0.01)
print(f"\n  Acceptance: firing <50% {'✅' if ok1 else '❌'} · less one-sided "
      f"{'✅' if ok2 else '❌'} · IC not worse {'✅' if ok3 else '❌'}")
print("  → " + ("ADOPT rl_v8" if ok1 and ok2 and ok3 else "do not adopt as-is; review the output"))
res.to_csv(f"{RESEARCH_DIR}/R8_rl_v8_replay.csv", index=False)
rp.to_csv(f"{RESEARCH_DIR}/R8_rl_v8_readings.csv", index=False)
print("\n  Save this output to the research log.")