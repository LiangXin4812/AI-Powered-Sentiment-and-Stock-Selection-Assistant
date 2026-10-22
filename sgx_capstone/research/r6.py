# ═══════════════════════════════════════════════════════════════════
# R6 — LONG-HISTORY TEST of the price signals (does short-term reversal
#      on these 15 SGX stocks hold up over years, not just 57 days?)
#
# Why: in R5 the only consistent pattern was short-term reversal (-ret5):
# IC +0.09 at h=1 and h=3, both halves positive, but IC_t only ~1.3 on
# 31-34 days — too few to tell. Prices exist for years, so the price part
# can be tested with real statistical power. Needs only SGX_STOCKS.
#
# Pre-declared signals (fixed before looking at the result):
#   REV5   -1 x past 5-day return   (short-term reversal)
#   REV1   -1 x past 1-day return
#   MOM60  past 60-day return, skipping the last 5 days (momentum)
# Target: next h-day return minus the 15-stock average (h = 1, 3, 5).
# ═══════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import yfinance as yf

START      = "2019-01-01"
OUR_WINDOW = "2026-06-11"          # start of the project's live data
HORIZONS6  = [1, 3, 5]
TICKERS6   = list(SGX_STOCKS.keys())

closes = {}
for t in TICKERS6:
    s = yf.Ticker(t).history(start=START, auto_adjust=True)["Close"]
    s.index = s.index.tz_localize(None).normalize()
    closes[t] = s
px = pd.DataFrame(closes).sort_index().dropna(how="all")
print(f"Prices: {px.index[0].date()} → {px.index[-1].date()}  "
      f"({len(px)} days, {px.notna().all(axis=1).sum()} with all 15 stocks)")

sig = {"REV5":  -(px / px.shift(5) - 1),
       "REV1":  -(px / px.shift(1) - 1),
       "MOM60": px.shift(5) / px.shift(65) - 1}


def nw_t(x, lags):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n)


rows = []
for h in HORIZONS6:
    fwd = px.shift(-h) / px - 1
    ex = fwd.sub(fwd.mean(axis=1), axis=0) * 100          # % excess vs the 15
    for name, S in sig.items():
        ic, spr, dates = [], [], []
        for d in px.index:
            a, b = S.loc[d], ex.loc[d]
            ok = a.notna() & b.notna()
            if ok.sum() < 12:
                continue
            ic.append(a[ok].rank().corr(b[ok].rank()))
            o = b[ok][a[ok].sort_values().index]
            spr.append(o.tail(5).mean() - o.head(5).mean())
            dates.append(d)
        ic, spr = pd.Series(ic, index=dates), pd.Series(spr, index=dates)
        by_y = ic.groupby(ic.index.year).mean()
        for period, m in [("2019-2025 before project", ic.index < OUR_WINDOW),
                          ("project window", ic.index >= OUR_WINDOW)]:
            i, s_ = ic[m], spr[m]
            if len(i) < 10:
                continue
            yrs = by_y[by_y.index < int(OUR_WINDOW[:4])] if "before" in period else by_y.iloc[-1:]
            rows.append({"signal": name, "h": h, "period": period, "days": len(i),
                         "IC": i.mean(), "IC_t": nw_t(i.values, max(h - 1, 1)),
                         "IC>0%": (i > 0).mean(),
                         "yrs_IC>0": f"{(yrs > 0).sum()}/{len(yrs)}",
                         # top-5 minus bottom-5, rebalanced every h days (no overlap)
                         "spread_per_rebal%": s_.iloc[::h].mean()})

r6 = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}")
print(r6.to_string(index=False))
try:
    r6.to_csv(f"{RESEARCH_DIR}/R6_long_history.csv", index=False)
except NameError:
    pass

print("""
  How to read this:
  • '2019-2025 before project' is the real test: ~1,700 days, so IC_t > 3 there
    is strong evidence, not luck. 'project window' shows whether it still held
    in our 57 days.
  • yrs_IC>0: in how many calendar years the signal pointed the right way.
  • spread_per_rebal%: top-5 minus bottom-5 per h-day period. Trading it costs
    roughly 0.3-0.6% per rebalance on SGX (commission, fees, bid-ask), so a
    spread below that is real information but not a profitable strategy alone.
  Save this table to the research log.""")