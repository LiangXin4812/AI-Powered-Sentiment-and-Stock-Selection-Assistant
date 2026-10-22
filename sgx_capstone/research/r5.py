# ═══════════════════════════════════════════════════════════════════
# R5 — PREDICTIVE VALUE TEST: reframe the target as RELATIVE performance
#
# Run AFTER the research cell (reuses: cache, ALL_DATES, POS, TICKERS,
# FEATURE_NAMES_T6, _q, _hdr, RESEARCH_DIR). Read-only.
#
# Why: the current target ("will this stock rise >1.09% in 3 days?") is
# dominated by the market's direction. The training period rose, September
# fell, so every model broke. An investor choosing among 15 SGX stocks needs
# RANKING: which stocks will beat the others. The market move cancels out of
# that question, so regime drift mostly disappears, and the naive baseline
# for "top third / middle / bottom third" is exactly 33% every day.
#
# Models compared (weekly walk-forward, no look-ahead, h = 1 and 3 days):
#   P   price-only    : 5d / 20d past return, 20d volatility (known anomalies)
#   F   T6 features   : what T6 uses today
#   PF  price + T6
#   T5  T5 rl_v7 net score, demeaned by day (no training at all)
#   T5s T5 stock-specific rules only (market-wide rules removed)
#
# Metrics (the ones a quant PM would ask for):
#   IC        mean daily Spearman correlation, prediction vs realised excess return
#   IC_t      Newey-West t-stat of the daily IC series (> 2 = unlikely to be luck)
#   IC>0%     share of days the ranking was right-way-round
#   spread    avg excess return of top-5 minus bottom-5 predicted stocks (%)
#   tercile   accuracy of top / mid / bottom-third calls (naive = 33.3%)
# ═══════════════════════════════════════════════════════════════════
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

R5_HORIZONS = [1, 3]
RIDGE_ALPHA = 10.0
_hdr("R5  RELATIVE-PERFORMANCE REFRAME")

# ── prices with 60 extra days of history for the past-return features ──
start = (datetime.strptime(ALL_DATES[0], "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
end = (datetime.strptime(ALL_DATES[-1], "%Y-%m-%d") + timedelta(days=14)).strftime("%Y-%m-%d")
PX = {}
for t in TICKERS:
    s = yf.Ticker(t).history(start=start, end=end)["Close"].sort_index()
    s.index = [x.strftime("%Y-%m-%d") for x in s.index]
    PX[t] = s

rows = []
for t in TICKERS:
    s = PX[t]
    idx, v = list(s.index), s.values
    lr = np.diff(np.log(v), prepend=np.nan)
    for j, d in enumerate(idx):
        if d not in POS or j < 20:
            continue
        r = {"ticker": t, "date": d,
             "px_ret5":  v[j] / v[j - 5] - 1,
             "px_ret20": v[j] / v[j - 20] - 1,
             "px_vol20": np.nanstd(lr[j - 19:j + 1])}
        for h in R5_HORIZONS:
            r[f"fwd{h}"] = (v[j + h] / v[j] - 1) * 100 if j + h < len(idx) else np.nan
        fd = cache.get((t, d))
        if fd:
            r.update({c: fd.get(c, 0) for c in FEATURE_NAMES_T6})
        rows.append(r)
panel = pd.DataFrame(rows)
PXF = ["px_ret5", "px_ret20", "px_vol20"]
FF = [c for c in FEATURE_NAMES_T6 if c in panel.columns]
panel[FF] = panel[FF].fillna(0)

# T5 scores (rl_v7) — total, stock-specific only
MARKET_WIDE = {"task1_political", "task3_macro", "task4_options"}
t5rows = []
for t, d, fr in _q("""
        SELECT DISTINCT ON (ticker, analysis_date::date)
               ticker, analysis_date::date::text, features_result
        FROM llm_analyses_audit
        WHERE (features_result::jsonb)->>'_rule_layer_version' = 'rl_v7'
        ORDER BY ticker, analysis_date::date, id DESC"""):
    fr = fr if isinstance(fr, dict) else json.loads(fr)
    net = stock = 0.0
    for f in fr.get("features", []):
        w = float(f.get("baseline_weight", 0.0)) * (1 if f.get("direction") == "bullish" else -1)
        net += w
        if f.get("source") not in MARKET_WIDE:
            stock += w
    t5rows.append({"ticker": t, "date": d, "t5_net": net, "t5_stock": stock})
_t5 = {(r["ticker"], r["date"]): r for r in t5rows}          # join without pd.merge
panel["t5_net"] = [(_t5.get((t, d)) or {}).get("t5_net", np.nan)
                   for t, d in zip(panel["ticker"], panel["date"])]
panel["t5_stock"] = [(_t5.get((t, d)) or {}).get("t5_stock", np.nan)
                     for t, d in zip(panel["ticker"], panel["date"])]

for h in R5_HORIZONS:
    # excess return vs the equal-weighted average of the 15 stocks that day
    panel[f"ex{h}"] = panel[f"fwd{h}"] - panel.groupby("date")[f"fwd{h}"].transform("mean")
# demean features cross-sectionally too: the model should learn "more X than
# the other stocks today", not the market-wide level of X
for c in PXF + FF:
    panel[c + "_x"] = panel[c] - panel.groupby("date")[c].transform("mean")
panel["week"] = pd.to_datetime(panel["date"]).dt.strftime("%G-W%V")
print(f"  Panel: {len(panel)} stock-days, {panel.date.nunique()} dates, "
      f"T5 coverage {panel.t5_net.notna().mean():.0%}")


def _nw_t(x, lags):
    """t-stat of the mean with Newey-West variance. With h-day returns the
    daily ICs overlap by h-1 days and are autocorrelated; a plain t-stat
    then overstates significance by up to sqrt(h)."""
    x = np.asarray(x, float); n = len(x); e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


def evaluate(pred, h, name):
    g = panel.assign(pred=pred).dropna(subset=["pred", f"ex{h}"])
    ics, spreads, terc = [], [], []
    for d, x in g.groupby("date"):
        if len(x) < 10 or x["pred"].nunique() < 3:
            continue
        ics.append(x["pred"].rank().corr(x[f"ex{h}"].rank()))
        o = x.sort_values("pred")
        spreads.append(o[f"ex{h}"].tail(5).mean() - o[f"ex{h}"].head(5).mean())
        n = len(x)                                           # terciles without pd.qcut
        pt = (np.argsort(np.argsort(x["pred"].values)) * 3) // n
        at = (np.argsort(np.argsort(x[f"ex{h}"].values)) * 3) // n
        terc.append((pt == at).mean())
    ics = np.array(ics)
    if len(ics) < 5:
        return None
    return {"h": h, "model": name, "days": len(ics),
            "IC": ics.mean(), "IC_t": _nw_t(ics, max(h - 1, 1)),
            "IC_1st_half": ics[:len(ics) // 2].mean(),     # stability: a real
            "IC_2nd_half": ics[len(ics) // 2:].mean(),     # signal holds in both
            "IC>0%": (ics > 0).mean(), "spread%": np.mean(spreads),
            "tercile_acc": np.mean(terc)}


def walk_forward_ridge(cols, h):
    """Weekly walk-forward ridge on cross-sectionally demeaned features."""
    out = pd.Series(np.nan, index=panel.index)
    for wk in sorted(panel["week"].unique()):
        te = panel[panel.week == wk]
        t0 = POS[te["date"].min()]
        tr = panel[(panel["date"].map(POS) + h < t0)].dropna(subset=[f"ex{h}"])
        if tr["date"].nunique() < MIN_TRAIN_DAYS:
            continue
        sc = StandardScaler().fit(tr[cols])
        m = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(tr[cols]), tr[f"ex{h}"])
        out.loc[te.index] = m.predict(sc.transform(te[cols]))
    return out


res = []
for h in R5_HORIZONS:
    P = [c + "_x" for c in PXF]
    F = [c + "_x" for c in FF]
    preds = {"P price-only":  walk_forward_ridge(P, h),
             "F T6-features": walk_forward_ridge(F, h),
             "PF price+T6":   walk_forward_ridge(P + F, h),
             "T5 net (demeaned)": panel["t5_net"] - panel.groupby("date")["t5_net"].transform("mean"),
             "T5s stock-only":    panel["t5_stock"],
             "reversal (-ret5)":  -panel["px_ret5"]}
    # keep the no-training scores on the same test days as the models
    tested = preds["P price-only"].notna()
    for name, p in preds.items():
        r = evaluate(p.where(tested), h, name)
        if r:
            res.append(r)
    # T5 on its full history as well (it has no training period)
    for name in ["T5 net (demeaned)", "T5s stock-only"]:
        r = evaluate(preds[name], h, name + " [all days]")
        if r:
            res.append(r)

r5 = pd.DataFrame(res)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(r5.to_string(index=False))
r5.to_csv(f"{RESEARCH_DIR}/R5_relative.csv", index=False)

print("""
  How to read this:
  • 12 rows are tested here, so on pure noise about one row can reach |IC_t| > 2
    by luck (verified on random data). Pre-registered primary tests:
    'PF price+T6' and 'T5 net (demeaned)'. A result counts only if IC_t > 2 AND
    both halves are positive AND h=1 and h=3 point the same way.
  • IC around 0.03-0.05 with IC_t > 2 is a real, usable edge in equity ranking;
    professional factor signals often live at IC 0.02-0.05.
  • spread% is the average excess return of 'top 5 minus bottom 5' per period,
    before costs. Positive and stable = something an investor can act on.
  • tercile_acc: random guessing scores 33.3%.
  • Compare PF vs P: if T6 features add IC on top of price-only, T1-T4 data
    carries information beyond price. Same for T5 vs reversal.
  Save this table to the research log.""")