"""A majority baseline reconstructed using only labels matured before each date."""
from collections import Counter

CLASS_ORDER = ("neutral", "bearish", "bullish")


def prior_majority_labels(history, evaluation):
    """No history uses the fixed neutral rule; later dates use prior matured labels.

    Rows have ISO date, maturity and actual fields. This uses recorded label
    maturity, not the report creation time, and never selects a class from the
    evaluation date's own outcome.
    """
    predictions = []
    for row in evaluation:
        known = Counter(x["actual"] for x in history
                        if x["maturity"] < row["date"])
        predictions.append(max(CLASS_ORDER, key=lambda c: known[c]) if known else "neutral")
    return predictions
