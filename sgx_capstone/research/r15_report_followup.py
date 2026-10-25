"""R15 - what followed the Daily Signal Report (23 Oct 2026).

Two questions an investor would ask of the report, answered on stored data only:

1. Do stocks marked worth a closer look (attention ELEVATED = the report's flag,
   or NOTABLE) move more over the next three sessions than the others?
2. Calls marked with a usable record (shown) are right more often than the rest
   (R10). Does that gain come from 'no big move' calls or from rise/fall calls?

Inputs
  * the report inputs saved by `check-dates`
    (outputs/date_check/<date>/T7_report_input_<date>.json; the copy used for
    the final report is kept in evidence/date_check_inputs/),
  * the matured three-session returns stored with R11
    (outputs/research/R11_predictions.csv, column y in percent),
  * the R10 replay of the stored champion forecasts
    (outputs/research/R10_suppression_replay.csv).

Output: outputs/research/R15_report_followup.csv, one row per group.
Exploratory: the flag rule was not designed to forecast size, and 18 report
dates are few; the result is reported, not used to tune anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _inputs(root: Path) -> list[Path]:
    for base in (root / "outputs" / "date_check", root / "evidence" / "date_check_inputs"):
        files = sorted(base.glob("**/T7_report_input_*.json"))
        if files:
            return files
    return []


def run(root: Path) -> pd.DataFrame:
    root = Path(root)
    res = root / "outputs" / "research"
    rows = []
    for f in _inputs(root):
        d = json.loads(f.read_text())
        for t, r in d["rows"].items():
            rows.append({"date": d["date"], "ticker": t, "attention": r["attention"]})
    att = pd.DataFrame(rows)
    out = []
    if len(att):
        y = (pd.read_csv(res / "R11_predictions.csv")[["date", "ticker", "y"]]
             .drop_duplicates(["date", "ticker"]))
        m = att.merge(y, on=["date", "ticker"], how="inner")
        m["abs3"] = m.y.abs()
        groups = {"flagged (ELEVATED)": m.attention == "ELEVATED",
                  "notable": m.attention == "NOTABLE",
                  "flagged or notable": m.attention.isin(["ELEVATED", "NOTABLE"]),
                  "routine": m.attention == "ROUTINE"}
        for g, sel in groups.items():
            out.append({"question": "attention vs next 3-session move", "group": g, "n": int(sel.sum()),
                        "measure": "mean absolute 3-session return %", "value": float(m.abs3[sel].mean()),
                        "dates": int(m.date.nunique()), "first": m.date.min(), "last": m.date.max()})
    r10 = pd.read_csv(res / "R10_suppression_replay.csv")
    hit = r10.pred == r10.actual
    shown, direction = r10.decision == "shown", r10.pred != "neutral"
    for g, sel in {"shown (record usable)": shown, "withheld": ~shown,
                   "shown, rise/fall call": shown & direction, "withheld, rise/fall call": ~shown & direction,
                   "shown, no-big-move call": shown & ~direction,
                   "withheld, no-big-move call": ~shown & ~direction}.items():
        out.append({"question": "forecast record split by call type", "group": g, "n": int(sel.sum()),
                    "measure": "share of calls right", "value": float(hit[sel].mean()),
                    "dates": int(r10.date.nunique()), "first": r10.date.min(), "last": r10.date.max()})
    out.append({"question": "forecast record split by call type", "group": "always 'no big move'",
                "n": int(len(r10)), "measure": "share of calls right", "value": float((r10.actual == "neutral").mean()),
                "dates": int(r10.date.nunique()), "first": r10.date.min(), "last": r10.date.max()})
    df = pd.DataFrame(out)
    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "R15_report_followup.csv", index=False)
    return df


def summary(df: pd.DataFrame) -> str:
    g = df.set_index("group")
    lines = []
    if "flagged (ELEVATED)" in g.index:
        lines.append(f"Next 3 sessions: flagged stocks moved {g.loc['flagged (ELEVATED)', 'value']:.2f}% on average "
                     f"(n {g.loc['flagged (ELEVATED)', 'n']}), flagged or notable {g.loc['flagged or notable', 'value']:.2f}%, "
                     f"routine {g.loc['routine', 'value']:.2f}% ({int(g.loc['routine', 'dates'])} report dates).")
    base = g.loc["always 'no big move'", "value"]
    lines.append(f"Shown calls right {g.loc['shown (record usable)', 'value']:.1%} vs withheld "
                 f"{g.loc['withheld', 'value']:.1%}; rise/fall calls {g.loc['shown, rise/fall call', 'value']:.1%} "
                 f"(n {g.loc['shown, rise/fall call', 'n']}) vs {g.loc['withheld, rise/fall call', 'value']:.1%}; "
                 f"no-big-move calls {g.loc['shown, no-big-move call', 'value']:.1%} vs "
                 f"{g.loc['withheld, no-big-move call', 'value']:.1%}; always 'no big move' {base:.1%}.")
    return "\n".join(lines)
