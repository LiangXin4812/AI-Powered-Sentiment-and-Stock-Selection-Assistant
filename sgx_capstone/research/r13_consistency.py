"""Rebuild the per-stock champion consistency table (R13) from R2 rows.

The pooled winner is chosen on these same evaluation dates, so this is a
retrospective diagnostic. A genuine prospective test needs a separate future
window for choosing the champion and evaluating stock-wise consistency.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score


def build(source: Path, destination: Path):
    rows = pd.read_csv(source)
    rows = rows[(rows.h == 3) & (rows.calib == "oof")].copy()
    models = sorted(rows.model.unique())
    if len(models) < 2:
        raise ValueError("Need at least two models on the 3-day target")
    keys = ["date", "ticker"]
    common = set.intersection(*(set(map(tuple, rows[rows.model == m][keys].values))
                                for m in models))
    rows = rows[rows[keys].apply(tuple, axis=1).isin(common)]
    labels = ["bearish", "bullish", "neutral"]
    pooled = {m: f1_score(x.actual, x.pred, labels=labels, average="macro", zero_division=0)
              for m, x in rows.groupby("model")}
    winner = max(models, key=lambda m: pooled[m])
    out = []
    for ticker, stock in rows.groupby("ticker"):
        scores = {}
        for model, frame in stock.groupby("model"):
            scores[model] = {
                "n": len(frame),
                "acc": (frame.pred == frame.actual).mean(),
                "macro_f1": f1_score(frame.actual, frame.pred, labels=labels,
                                     average="macro", zero_division=0),
                "prior_rule_acc": (frame.naive == frame.actual).mean(),
            }
        best_other = max(scores[m]["macro_f1"] for m in models if m != winner)
        out.append({"ticker": ticker, "pooled_winner": winner,
                    "winner_leads_f1": scores[winner]["macro_f1"] > best_other,
                    **{f"{m}_{k}": v for m in models for k, v in scores[m].items()}})
    result = pd.DataFrame(out).sort_values("ticker")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Pooled winner {winner}: {sum(result.winner_leads_f1)}/{len(result)} "
          "stocks lead on macro F1 (retrospective, same dates).")
    return result


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2] / "outputs" / "research"
    build(base / "walk_forward_predictions.csv", base / "R13_champion_consistency.csv")
