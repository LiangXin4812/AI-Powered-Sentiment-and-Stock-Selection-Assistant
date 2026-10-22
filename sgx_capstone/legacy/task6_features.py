# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE: extract_features_row  (copied verbatim from Task 5 Part A v6)
#
# WHY THIS CELL EXISTS
#   Task 6 Part A and Task 5 Part B both call extract_features_row, which is
#   defined inside Task 5 Part A. After a kernel restart that means running
#   all of T5 Part A (GridSearchCV etc.) just to get one function. This cell
#   provides it in ~1 second.
#
# ⚠️  DRIFT WARNING — read before editing
#   This is a verbatim copy of Part A v6's Step 0, including its constants
#   (BULLISH_EVENTS, BEARISH_EVENTS, REGIME_ENCODE). Two copies of a shared
#   extractor is exactly the pattern we avoided all week; it is tolerated
#   here only as a session convenience. Rules:
#     1. NEVER edit the function here. Edit Part A, then re-copy.
#     2. If Part A's extractor changes (v7+), delete this cell or re-copy it
#        the same day.
#   The proper fix is moving the extractor into src/features.py that both
#   Parts import — queued for the deployment refactor.
#
# Prereqs: Setup cell only (SUPABASE_ENGINE, text).
# ═══════════════════════════════════════════════════════════════════════════

from sqlalchemy import text

# ── Constants the extractor needs (verbatim from Part A v6) ─────────────────

BULLISH_EVENTS = {"upgrade", "earnings", "dividend", "ma"}    # ma = merger/acquisition
BEARISH_EVENTS = {"warning", "downgrade", "layoff"}

REGIME_ENCODE = {
    "Recessionary":                    -2,   # Yield curve inverted
    "Risk-Off":                        -1,   # VIX > 30
    "Neutral":                          0,
    "Risk-On":                         +1,   # VIX < 15
    "High Rate / Elevated Inflation":  -1,
    "Low Rate / Low Inflation":        +1,
}


def extract_features_row(ticker, date, return_baseline_sentiment=False):
    """
    Extract features for one (ticker, date) pair.

    Returns a features dict, or None if no technical data exists.
    With return_baseline_sentiment=True, returns (features, baseline) where
    baseline is a coarse news-direction label used by Part B's SHAP layer.
    """
    with SUPABASE_ENGINE.connect() as conn:
        # ── Task 1: company news aggregates (ticker-specific) ──
        news = conn.execute(text("""
            SELECT COUNT(*),
                   SUM(CASE WHEN sentiment='bullish' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN sentiment='bearish' THEN 1 ELSE 0 END),
                   AVG(sentiment_score)
            FROM news_articles WHERE ticker=:t AND date=:d AND category='company'
        """), {"t": ticker, "d": date}).fetchone()

        # ── Task 1: political news — volume and average direction (market-wide)
        pol = conn.execute(text("""
            SELECT COUNT(*), AVG(sentiment_score)
            FROM news_articles WHERE ticker='POLITICAL' AND date=:d
        """), {"d": date}).fetchone()
        pol_count     = pol[0] or 0
        pol_direction = float(pol[1]) if pol[1] is not None else 0.0

        # ── Task 1: event counts. Counts rather than binary flags, because the
        # binary version fired on ~92% of samples and effectively encoded
        # "did the ETL run today" instead of "was there news".
        event_counts = conn.execute(text("""
            SELECT event_flag, COUNT(*)
            FROM news_articles
            WHERE ticker=:t AND date=:d AND category='company' AND event_flag != 'none'
            GROUP BY event_flag
        """), {"t": ticker, "d": date}).fetchall()

        # ── Task 2: technical signals (ticker-specific) ──
        tech = conn.execute(text("""
            SELECT rsi, macd_signal, bb_position, vwap_position, weekly_signal
            FROM technical_signals WHERE ticker=:t AND date=:d
        """), {"t": ticker, "d": date}).fetchone()

        # ── Task 3: macro indicators (market-wide) ──
        macro = conn.execute(text("""
            SELECT vix, us_yield_spread, macro_regime
            FROM macro_indicators WHERE date=:d
        """), {"d": date}).fetchone()

        # ── Task 4: options positioning (market-wide) ──
        opt = conn.execute(text("""
            SELECT options_signal, pc_volume_ratio
            FROM market_options WHERE date=:d
        """), {"d": date}).fetchone()

    # No technical data → this sample cannot be featurised
    if tech is None:
        return (None, None) if return_baseline_sentiment else None

    n_total = news[0] or 0
    n_bull  = news[1] or 0
    n_bear  = news[2] or 0
    event_set     = {ef for ef, _ in event_counts}
    n_bull_events = sum(cnt for ef, cnt in event_counts if ef in BULLISH_EVENTS)
    n_bear_events = sum(cnt for ef, cnt in event_counts if ef in BEARISH_EVENTS)

    news_bullish_ratio = n_bull / n_total if n_total > 0 else 0.0
    news_bearish_ratio = n_bear / n_total if n_total > 0 else 0.0

    # Missing-data defaults use neutral values so absence never injects signal:
    # ratios 0.0, RSI 50 (mid-range), VIX 20 (long-run average),
    # yield spread 0.5 (normal positive slope), P/C 1.0 (balanced).
    features = {
        # ── News ──
        "news_bullish_ratio":     news_bullish_ratio,
        "news_bearish_ratio":     news_bearish_ratio,
        "avg_sentiment":          float(news[3]) if news[3] else 0.0,
        "n_articles":             float(n_total),
        "n_political":            float(pol_count),      # market-wide
        "political_direction":    pol_direction,          # market-wide
        "has_bullish_events":     1 if any(e in BULLISH_EVENTS for e in event_set) else 0,
        "has_bearish_events":     1 if any(e in BEARISH_EVENTS for e in event_set) else 0,
        "n_bullish_events":       float(n_bull_events),
        "n_bearish_events":       float(n_bear_events),
        # ── Technical ──
        "rsi":                    float(tech[0]) if tech[0] else 50.0,
        "macd_bullish_crossover": 1 if tech[1] == "bullish_crossover" else 0,
        "macd_bearish_crossover": 1 if tech[1] == "bearish_crossover" else 0,
        "bb_above_upper":         1 if tech[2] == "above_upper" else 0,
        "bb_below_lower":         1 if tech[2] == "below_lower" else 0,
        "vwap_above":             1 if tech[3] == "above" else 0,
        "weekly_bullish":         1 if tech[4] == "bullish" else 0,
        # ── Macro (market-wide) ──
        "regime_encoded":         REGIME_ENCODE.get(macro[2] if macro else "Neutral", 0),
        "vix":                    float(macro[0]) if macro and macro[0] else 20.0,
        "yield_spread":           float(macro[1]) if macro and macro[1] else 0.5,
        # ── Options (market-wide) ──
        "options_bullish":        1 if opt and opt[0] == "bullish" else 0,
        "options_bearish":        1 if opt and opt[0] == "bearish" else 0,
        "pc_vol_ratio":           float(opt[1]) if opt and opt[1] else 1.0,
    }

    if return_baseline_sentiment:
        # Coarse news-direction label for Part B's SHAP cross-check.
        d = news_bullish_ratio - news_bearish_ratio
        baseline = "bullish" if d > 0.15 else "bearish" if d < -0.15 else "neutral"
        return features, baseline

    return features


