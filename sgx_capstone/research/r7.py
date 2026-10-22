# ═══════════════════════════════════════════════════════════════════
# R7 — T5 RULE AUDIT: which rules cause the permanent bullish tilt, and
#      does any single rule carry information?
#
# R4 showed the tilt is NOT mainly market-wide: mean net +0.381 =
# stock-specific +0.289 + market-wide +0.092, and the stock-specific part
# stays at +0.28..+0.31 every week while the market part fell to -0.09.
# So some stock-specific rules fire bullish far more than bearish.
# Run after the research cell (reuses _q, FWD, TICKERS, RESEARCH_DIR).
# ═══════════════════════════════════════════════════════════════════
import json
import numpy as np
import pandas as pd

# next-day excess return vs the 15-stock average
_r1 = {}
for (t, d, h), r in FWD.items():
    if h == 1:
        _r1.setdefault(d, {})[t] = r
EX1 = {(t, d): r - np.mean(list(v.values()))
       for d, v in _r1.items() if len(v) >= 12 for t, r in v.items()}

rec = []
for t, d, fr in _q("""
        SELECT DISTINCT ON (ticker, analysis_date::date)
               ticker, analysis_date::date::text, features_result
        FROM llm_analyses_audit
        WHERE (features_result::jsonb)->>'_rule_layer_version' = 'rl_v7'
        ORDER BY ticker, analysis_date::date, id DESC"""):
    fr = fr if isinstance(fr, dict) else json.loads(fr)
    for f in fr.get("features", []):
        sgn = 1 if f.get("direction") == "bullish" else -1
        rec.append({"factor": f.get("factor", "?"), "source": f.get("source", "?"),
                    "signed_w": sgn * float(f.get("baseline_weight", 0.0)), "dir": sgn,
                    "ex1": EX1.get((t, d))})
fx = pd.DataFrame(rec)
n_read = len({(r[0], r[1]) for r in _q("""
        SELECT DISTINCT ticker, analysis_date::date FROM llm_analyses_audit
        WHERE (features_result::jsonb)->>'_rule_layer_version' = 'rl_v7'""")})

rows = []
for (fac, src), g in fx.groupby(["factor", "source"]):
    e = g.dropna(subset=["ex1"])
    rows.append({"factor": fac[:48], "source": src.replace("task", "T"),
                 "fired%": len(g) / n_read,
                 "bull_share": (g["dir"] > 0).mean(),
                 "avg_signed_w": g["signed_w"].mean(),
                 "tilt_contrib": g["signed_w"].sum() / n_read,   # adds this to the average net score
                 "n_graded": len(e),
                 # did the rule's direction match the next-day EXCESS return?
                 "dir_hit%": (np.sign(e["ex1"]) == e["dir"]).mean() if len(e) else np.nan,
                 "avg_ex1_signed%": (e["ex1"] * e["dir"]).mean() if len(e) else np.nan})
aud = pd.DataFrame(rows).sort_values("tilt_contrib", ascending=False)
pd.set_option("display.width", 220, "display.max_rows", 200,
              "display.float_format", lambda x: f"{x:.3f}")
print(f"R7  T5 RULE AUDIT — {n_read} rl_v7 readings, {len(aud)} distinct rules")
print(aud.to_string(index=False))
print(f"\n  Sum of tilt_contrib = {aud['tilt_contrib'].sum():+.3f} "
      f"(should equal the mean net score in R4, +0.381)")
aud.to_csv(f"{RESEARCH_DIR}/R7_t5_rule_audit.csv", index=False)
print("""
  How to read this:
  • tilt_contrib: how much this rule adds to the AVERAGE net score. The top
    rows are where the permanent bullish tilt comes from.
  • bull_share = 1.00 means the rule can only ever fire bullish (no mirror
    bearish rule exists) — a structural cause of zero bearish calls.
  • dir_hit% / avg_ex1_signed%: > 50% / > 0 means the rule's direction beat
    the other stocks the next day. Rules with many firings and dir_hit well
    below 50% are candidates to drop or flip.
  Save this table to the research log.""")