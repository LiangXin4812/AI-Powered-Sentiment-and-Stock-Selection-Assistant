"""Which Task 5 rules fired, re-run offline from stored feature rows (R8 detail).

The offline sample of the Daily Signal Report has no database, so it cannot
read the fired rules from llm_analyses_audit as the live report does. This
script re-runs the production rule layer (task5_runtime.compute_feature_
contributions, rl_v8 settings, pinned v6 weights) on the stored Task 1-4
feature rows in tests/fixtures/feature_rows.csv and keeps a stock-day only if
the re-run reproduces the stored rl_v8 net score in R8_rl_v8_readings.csv to
within rounding (0.0015).

Two inputs are not in the feature rows and are reconstructed:
  * event-rule firing is taken from the stored R8 flag (evt_fired_v8);
  * the rows store weekly_bullish only, so when it is 0 the weekly signal is
    tried as "neutral" and as "bearish" and kept only if exactly one of the
    two reproduces the stored score (the two differ by a full rule weight).
Stock-days that do not reproduce are written with reproduced=False and are
not used by the report.

    python -m sgx_capstone.research.r8_rule_detail
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

TOL = 0.0015


def _load_rule_layer(root: Path):
    from sgx_capstone.configuration import apply_config, load_config
    from sgx_capstone.data_access import load_settings
    from sgx_capstone.__main__ import _execute

    cfg, sha = load_config()
    settings = load_settings(root)
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = 0
    conn.execute.return_value.fetchone.return_value = None
    conn.execute.return_value.fetchall.return_value = []
    engine = MagicMock()                       # no database is touched
    engine.connect.return_value.__enter__.return_value = conn
    ns = dict(CFG=cfg, CONFIG_SHA=sha,
              CONFIG={"technical": settings["solution"]["technical"],
                      "storage": settings["system"]["storage"]},
              SUPABASE_ENGINE=engine, SGX_STOCKS=settings["sgx"]["tickers"])
    apply_config(cfg, ns)
    ns["apply_config"] = lambda c: apply_config(c, ns)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for name in ("task6_features", "task5_runtime"):
            _execute(name, ns)
            apply_config(cfg, ns)
    sectors = {t: v["sector"] for t, v in settings["sgx"]["tickers"].items()}
    return ns, sectors


def _context(r, sector, evt_fired, weekly, regime_decode):
    n = int(round(r.n_articles))
    nb, nr = int(round(r.news_bullish_ratio * n)), int(round(r.news_bearish_ratio * n))
    events = (["earnings"] if r.n_bullish_events > 0 else []) + \
             (["warning"] if r.n_bearish_events > 0 else [])
    rsi = float(r.rsi)
    return {
        "sector": sector,
        "news": {"company_articles": n, "bullish_count": nb, "bearish_count": nr,
                 "neutral_count": n - nb - nr, "avg_sentiment": float(r.avg_sentiment),
                 "events": events,
                 "event_counts": {"bull_today": float(r.n_bullish_events),
                                  "bull_trail": float(r.n_bullish_events) - (1.0 if evt_fired else 0.0),
                                  "bear_today": float(r.n_bearish_events),
                                  "bear_trail": float(r.n_bearish_events), "trail_days": 20}},
        "political": {"total": int(r.n_political), "avg_sentiment": float(r.political_direction)},
        "technical": {"rsi": rsi,
                      "rsi_signal": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
                      "macd_value": 0.0,
                      "macd_signal": "bullish_crossover" if r.macd_bullish_crossover else
                                     "bearish_crossover" if r.macd_bearish_crossover else "neutral",
                      "bb_position": "above_upper" if r.bb_above_upper else
                                     "below_lower" if r.bb_below_lower else "within",
                      "vwap_position": "above" if r.vwap_above else "below",
                      "weekly_signal": weekly},
        "macro": {"regime": regime_decode.get(int(r.regime_encoded), "Neutral"),
                  "vix": float(r.vix), "yield_spread": float(r.yield_spread)},
        "options": {"signal": "bullish" if r.options_bullish else
                              "bearish" if r.options_bearish else "neutral",
                    "pc_volume_ratio": float(r.pc_vol_ratio)},
    }


def build(root: Path, destination: Path) -> pd.DataFrame:
    ns, sectors = _load_rule_layer(root)
    decode = {v: k for k, v in ns["REGIME_ENCODE"].items()}
    fx = pd.read_csv(root / "tests" / "fixtures" / "feature_rows.csv")
    r8 = pd.read_csv(root / "outputs" / "research" / "R8_rl_v8_readings.csv")
    out = []
    for _, r in fx.iterrows():
        s = r8[(r8.ticker == r.ticker) & (r8.date == r.date)]
        if s.empty:
            continue
        s = s.iloc[0]
        stored = round(float(s.net_rl_v8), 3)
        options = ["bullish"] if r.weekly_bullish else ["neutral", "bearish"]
        hits = []
        for weekly in options:
            res = ns["compute_feature_contributions"](
                _context(r, sectors[r.ticker], bool(s.evt_fired_v8), weekly, decode))
            if abs(res["score_breakdown"]["net_score"] - stored) < TOL:
                hits.append((weekly, res))
        ok = len(hits) == 1
        weekly, res = hits[0] if ok else (None, None)
        feats = res["features"] if ok else [{}]
        for f in feats:
            out.append({"ticker": r.ticker, "date": r.date, "reproduced": ok,
                        "net_stored": stored,
                        "net_offline": res["score_breakdown"]["net_score"] if ok else None,
                        "weekly_used": weekly, "factor": f.get("factor"),
                        "direction": f.get("direction"), "source": f.get("source"),
                        "weight": f.get("baseline_weight"), "decisive": bool(f.get("flips_call"))})
    result = pd.DataFrame(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    days = result.drop_duplicates(["ticker", "date"])
    print(f"R8 rule detail: {int(days.reproduced.sum())} of {len(days)} stock-days reproduce "
          f"the stored rl_v8 net score -> {destination.name}")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    build(root, root / "outputs" / "research" / "R8_rl_v8_rule_detail.csv")
