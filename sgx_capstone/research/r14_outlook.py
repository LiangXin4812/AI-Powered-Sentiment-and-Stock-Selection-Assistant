"""R14 - what can be said about the next three sessions, and how often has
each kind of statement held up? (Exploratory; not pre-registered.)

Input: outputs/research/walk_forward_predictions.csv, the stored weekly
walk-forward predictions with out-of-fold calibration (same 21 features and
model families as production). Nothing is refitted; stored predictions are
scored against realised returns.

Event definition used for every rate below (stated in every output row):
    a "move" is an absolute three-session return above 1.0%  (|ret| > 1.0)
The models' own label band varies by training window (half the s.d. of
training returns, about 1.0-1.1%), so their "neutral" class is not exactly
this event. Rates on the label-band definition are kept as a secondary
column for comparison; the calibration table uses the label band because
that is the event the probabilities were trained for.

Statements tested:
  S1 Typical range. +/- s * sqrt(3), s = s.d. of the last 20 daily returns
     available before the date. In this file daily returns exist only on
     stored walk-forward dates (34 of the 42 trading days from 21 Jul to
     17 Sep), so the 20 returns span more than 20 trading days. The live
     report uses contiguous closing prices instead.
  S2 Quieter than most. Each day, the five of fifteen stocks with the lowest
     model score 1 - P(neutral). Compared with the other ten, by half, by
     date (strictly lower move rate; ties reported), on non-overlapping
     windows (every third date), and against the same rule built from
     20-day volatility.
  S3 Direction. Among stocks with a move, does P(bullish) - P(bearish)
     separate rises from falls?

Honesty notes: defined after R2 (post hoc); 31 overlapping dates; the
"at least 15 new dates" bar is this project's own check, not a statistical
certification.
"""
from __future__ import annotations

import json
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = 3
N_QUIET = 5
VOL_WINDOW = 20
VOL_MIN = 5
EVENT_PCT = 1.0
EVENT = f"|three-session return| > {EVENT_PCT:.1f}%"
FROZEN_AT = "2026-10-16"
# Exchange sessions in the stored period (SGX closed on Mon 10 Aug 2026,
# National Day observed); used to date when a three-session outcome matured.
SGX_HOLIDAYS = ("2026-08-10",)
SESSION = pd.offsets.CustomBusinessDay(holidays=list(SGX_HOLIDAYS))
ACCEPTANCE = {
    "event_definition": EVENT,
    "min_new_dates": 15,
    "quiet_move_rate_below_rest_by": 0.10,
    "quiet_group_strictly_lower_on_share_of_dates": 0.60,
    "range_coverage_between": [0.60, 0.80],
}


def _p_greater(k: int, n: int) -> float:
    """One-sided exact sign test P(X >= k | n, 0.5)."""
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else float("nan")


def _prior_vol(wf: pd.DataFrame) -> pd.DataFrame:
    d1 = (wf[(wf.h == 1) & (wf.calib == "oof") & (wf.model == "RandomF")]
          [["ticker", "date", "ret"]].sort_values(["ticker", "date"]).copy())
    # The h=1 return stored on date t runs from t to the next close, so the
    # return known at the close of t is the one stored on the previous date.
    d1["r_known"] = d1.groupby("ticker").ret.shift(1)
    d1["vol"] = d1.groupby("ticker").r_known.transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=VOL_MIN).std())
    return d1[["ticker", "date", "vol"]]


def _quiet(x: pd.DataFrame, score: str, flag: str = "move") -> dict:
    x = x.copy()
    x["quiet"] = x.groupby("date")[score].rank(method="first") <= N_QUIET
    dates = sorted(x.date.unique())
    half = dates[len(dates) // 2]
    per = x.groupby("date")[["quiet", flag]].apply(
        lambda t: (t.loc[t.quiet, flag].mean(), t.loc[~t.quiet, flag].mean()))
    strict = int(sum(a < b for a, b in per))
    ties = int(sum(a == b for a, b in per))
    nov_dates = dates[::H]
    nov = x[x.date.isin(nov_dates)]
    per_n = per[per.index.isin(nov_dates)]
    strict_n = int(sum(a < b for a, b in per_n))
    ties_n = int(sum(a == b for a, b in per_n))
    q, r = x[x.quiet], x[~x.quiet]
    f, s = x[x.date < half], x[x.date >= half]
    return {
        "n": int(len(x)), "dates": len(dates),
        "quiet_moves": int(q[flag].sum()), "quiet_n": int(len(q)),
        "quiet_move_rate": float(q[flag].mean()),
        "rest_moves": int(r[flag].sum()), "rest_n": int(len(r)),
        "rest_move_rate": float(r[flag].mean()),
        "quiet_mean_abs_ret": float(q.ret.abs().mean()),
        "rest_mean_abs_ret": float(r.ret.abs().mean()),
        "quiet_move_rate_first_half": float(f[f.quiet][flag].mean()),
        "rest_move_rate_first_half": float(f[~f.quiet][flag].mean()),
        "quiet_move_rate_second_half": float(s[s.quiet][flag].mean()),
        "rest_move_rate_second_half": float(s[~s.quiet][flag].mean()),
        "dates_quiet_strictly_lower": strict, "dates_tied": ties,
        "p_dates_one_sided": _p_greater(strict, len(per) - ties),
        "nonoverlap_dates": len(nov_dates),
        "nonoverlap_quiet_move_rate": float(nov[nov.quiet][flag].mean()),
        "nonoverlap_rest_move_rate": float(nov[~nov.quiet][flag].mean()),
        "nonoverlap_dates_quiet_strictly_lower": strict_n,
        "p_nonoverlap_one_sided": _p_greater(strict_n, len(per_n) - ties_n),
    }


def run(root: Path, as_of: str | None = None, write: bool = True) -> dict:
    """R14 on all stored dates (the frozen record, written to evidence/), or,
    with as_of, only on forecast dates whose three-session outcome had matured
    by that close: the record as it was known on a report date (not written)."""
    root = Path(root)
    wf = pd.read_csv(root / "outputs" / "research" / "walk_forward_predictions.csv")
    if as_of is not None:
        due = pd.to_datetime(wf.date).map(lambda d: d + H * SESSION)
        wf = wf[(wf.h != H) | (due <= pd.Timestamp(as_of))].copy()
    vol = _prior_vol(wf)
    rows, calib_rows, by_model, evidence = [], [], {}, {}

    for model in ("RandomF", "XGBoost", "LogReg"):
        x = wf[(wf.h == H) & (wf.calib == "oof") & (wf.model == model)].copy()
        x["move"] = (x.ret.abs() > EVENT_PCT).astype(int)
        x["move_label_band"] = (x.actual != "neutral").astype(int)
        x["p_move"] = 1 - x.p_neu
        x = x.merge(vol, on=["ticker", "date"], how="left")
        moved = x[x.move == 1]
        lean = np.sign(x.p_bull - x.p_bear)
        st = _quiet(x, "p_move")
        lb = _quiet(x, "p_move", "move_label_band")
        row = {"model": model, "statement": "S2 quieter than most (model)",
               "event_definition": EVENT, "horizon_sessions": H,
               "auc_move": roc_auc_score(x.move, x.p_move),
               "base_move_rate": float(x.move.mean()),
               "auc_direction_among_moves": roc_auc_score(
                   (moved.ret > 0).astype(int), moved.p_bull - moved.p_bear),
               "lean_sign_hit": float(((lean > 0) & (x.ret > 0) | (lean < 0) & (x.ret < 0)).mean()),
               "share_up": float((x.ret > 0).mean()),
               **st,
               "label_band_quiet_move_rate": lb["quiet_move_rate"],
               "label_band_rest_move_rate": lb["rest_move_rate"]}
        rows.append(row)
        by_model[model] = {k: row[k] for k in (
            "quiet_move_rate", "rest_move_rate", "dates_quiet_strictly_lower", "dates",
            "auc_move", "auc_direction_among_moves", "lean_sign_hit")}
        by_model[model]["quiet_record"] = bool(
            row["rest_move_rate"] - row["quiet_move_rate"] >= ACCEPTANCE["quiet_move_rate_below_rest_by"]
            and row["dates_quiet_strictly_lower"] / row["dates"]
            >= ACCEPTANCE["quiet_group_strictly_lower_on_share_of_dates"])

        if model == "RandomF":
            v = x.dropna(subset=["vol"]).copy()
            vs = _quiet(v, "vol")
            ms = _quiet(v, "p_move")
            v["mq"] = v.groupby("date").p_move.rank(method="first") <= N_QUIET
            v["vq"] = v.groupby("date").vol.rank(method="first") <= N_QUIET
            both = v[v.mq & v.vq]
            vrow = {"model": "volatility only", "statement": "S2 quieter than most (20-day volatility)",
                    "event_definition": EVENT, "horizon_sessions": H,
                    "auc_move": roc_auc_score(v.move, v.vol),
                    "auc_move_model_same_rows": roc_auc_score(v.move, v.p_move),
                    "model_quiet_move_rate_same_rows": ms["quiet_move_rate"],
                    "model_rest_move_rate_same_rows": ms["rest_move_rate"],
                    "base_move_rate": float(v.move.mean()), **vs,
                    "both_quiet_n": int(len(both)), "both_quiet_moves": int(both.move.sum()),
                    "both_quiet_move_rate": float(both.move.mean())}
            rows.append(vrow)

            band = v.vol * sqrt(H)
            cover = v.ret.abs() <= band
            half = sorted(v.date.unique())[len(v.date.unique()) // 2]
            rrow = {"model": "volatility only", "statement": "S1 typical range +/- s20*sqrt(3)",
                    "event_definition": "realised three-session return inside the range",
                    "horizon_sessions": H, "n": int(len(v)), "dates": int(v.date.nunique()),
                    "inside": int(cover.sum()), "range_coverage": float(cover.mean()),
                    "range_coverage_first_half": float(cover[v.date < half].mean()),
                    "range_coverage_second_half": float(cover[v.date >= half].mean()),
                    "median_half_width_pct": float(band.median()),
                    "median_abs_ret_pct": float(v.ret.abs().median()),
                    "note": "daily returns only on stored walk-forward dates; window spans more than 20 trading days"}
            rows.append(rrow)

            # Calibration of the stated probability against the event it was
            # trained for (the label band), not the fixed 1% event.
            x["bin"] = pd.cut(x.p_move, [0, .40, .45, .50, .55, .60, 1])
            for b, g in x.groupby("bin", observed=True):
                calib_rows.append({"model": model, "event_definition": "model label band (neutral class)",
                                   "p_move_bin": str(b), "n": len(g),
                                   "stated_mean": float(g.p_move.mean()),
                                   "observed_move_rate": float(g.move_label_band.mean())})
            dates = sorted(x.date.unique())
            evidence = {
                "source": "outputs/research/walk_forward_predictions.csv (h=3, out-of-fold sigmoid calibration, RandomF)",
                "event_definition": EVENT, "horizon_sessions": H,
                "window": f"{dates[0]} to {dates[-1]}", "frozen_at": FROZEN_AT,
                "pre_registered": False, "acceptance_for_new_dates": ACCEPTANCE,
                "n_stock_days": row["n"], "n_dates": row["dates"],
                "range": {k: rrow[k] for k in ("n", "inside", "range_coverage",
                                                "range_coverage_first_half",
                                                "range_coverage_second_half", "note")},
                "quiet": {k: row[k] for k in (
                    "quiet_moves", "quiet_n", "quiet_move_rate", "rest_moves", "rest_n",
                    "rest_move_rate", "quiet_mean_abs_ret", "rest_mean_abs_ret",
                    "quiet_move_rate_first_half", "rest_move_rate_first_half",
                    "quiet_move_rate_second_half", "rest_move_rate_second_half",
                    "dates_quiet_strictly_lower", "dates_tied", "dates", "p_dates_one_sided",
                    "nonoverlap_dates", "nonoverlap_quiet_move_rate", "nonoverlap_rest_move_rate",
                    "nonoverlap_dates_quiet_strictly_lower", "p_nonoverlap_one_sided")},
                "quiet_both_rules": {"n": vrow["both_quiet_n"], "moves": vrow["both_quiet_moves"],
                                     "move_rate": vrow["both_quiet_move_rate"]},
                "volatility_rule": {"auc_move": vrow["auc_move"],
                                    "auc_move_model_same_rows": vrow["auc_move_model_same_rows"],
                                    "quiet_move_rate": vrow["quiet_move_rate"],
                                    "rest_move_rate": vrow["rest_move_rate"],
                                    "model_quiet_move_rate_same_rows": vrow["model_quiet_move_rate_same_rows"]},
                "move_auc": row["auc_move"],
                "direction": {"auc_among_moves": row["auc_direction_among_moves"],
                              "lean_sign_hit": row["lean_sign_hit"], "share_up": row["share_up"]},
            }

    evidence["by_model"] = by_model
    if as_of is not None:
        evidence["as_of"] = str(as_of)
        evidence["frozen_at"] = None
        return evidence
    if not write:
        return evidence
    out = root / "outputs" / "research"
    pd.DataFrame(rows).to_csv(out / "R14_outlook_reliability.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out / "R14_move_calibration.csv", index=False)
    (root / "evidence").mkdir(exist_ok=True)
    (root / "evidence" / "outlook_reliability.json").write_text(json.dumps(evidence, indent=2, default=float))
    return evidence


if __name__ == "__main__":
    import sys
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    ev = run(base)
    q = ev["quiet"]
    print(f"R14 ({ev['event_definition']}) on {ev['n_stock_days']} stock-days, {ev['n_dates']} dates")
    print(f"  quiet five {q['quiet_moves']}/{q['quiet_n']} = {q['quiet_move_rate']:.1%}; "
          f"rest {q['rest_moves']}/{q['rest_n']} = {q['rest_move_rate']:.1%}; "
          f"strictly lower on {q['dates_quiet_strictly_lower']}/{q['dates']} dates ({q['dates_tied']} ties)")
    print(f"  range {ev['range']['inside']}/{ev['range']['n']} = {ev['range']['range_coverage']:.1%}; "
          f"direction AUC {ev['direction']['auc_among_moves']:.3f}")
