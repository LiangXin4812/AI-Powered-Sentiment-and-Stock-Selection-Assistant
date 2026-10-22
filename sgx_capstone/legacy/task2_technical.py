# ═══════════════════════════════════════════════════════════════════
# SGX Decision Support System
# TASK 2 — Technical & Momentum Analysis Engine
#
# Purpose:
#   Compute daily technical indicators for 15 SGX blue-chip stocks
#   and aggregate them into an overall directional signal
#   (bullish / neutral / bearish) per stock.
#
# Pipeline steps:
# 1. Fetch 90-day daily OHLCV from Yahoo Finance (with timeout + retry)
# 2. Calculate technical indicators using pandas-ta:
#      RSI(14), MACD(12/26/9), Bollinger Bands(20, 2σ),
#      ATR(14), custom VWAP proxy, Volume SMA(20)
# 3. Detect signal events:
#      RSI overbought/oversold, MACD crossovers, BB breakouts,
#      volume anomalies
# 4. Multi-timeframe: confirm signals on weekly chart
# 5. Aggregate to one overall signal per stock via weighted scoring
# 6. Store results in technical_signals table
#      ⭐ CHANGED: Supabase primary, SQLite fallback (was dual-write)
#
# Resilience notes (added after 6/29 production incident):
#   - All yfinance calls wrapped with timeout=15s + retry (max 3x)
#     Reason: Yahoo's 2026 anti-scraping changes caused indefinite hangs
#   - Column lookups use prefix matching (find_col helper)
#     Reason: pandas-ta changed Bollinger column names across versions
#     e.g. "BBU_20_2.0" → "BBU_20_2.0_2.0" in newer releases
#   - Defensive pd.isna() checks throughout to handle NaN in early
#     rolling-window periods (first 20 days RSI/BB are NaN by design)
#   - Per-stock try/except: single-stock failure doesn't abort pipeline
# ═══════════════════════════════════════════════════════════════════

import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import sqlite3
import time
from datetime import datetime
from sqlalchemy import text


# ── Step 1: Fetch OHLCV data ──────────────────────────
def fetch_ohlcv(ticker, period="90d", interval="1d", max_retries=3, timeout=15):
    """
    Fetch historical OHLCV data from Yahoo Finance with resilience.

    90 days of daily data is sufficient for all indicators:
      - RSI(14):        needs 14+ days warmup
      - MACD(26):       needs 26+ days warmup
      - Bollinger(20):  needs 20+ days warmup
      - Weekly RSI(14): calls separately with period="1y" interval="1wk"
    90 days gives ~64 valid signal rows after warmup — plenty of history.

    SGX tickers must have .SI suffix (e.g., D05.SI for DBS).

    Resilience:
      Yahoo Finance occasionally hangs indefinitely due to anti-scraping
      measures (introduced 2026). Wrapping with timeout + retry prevents
      the pipeline from blocking on a single bad request.

    Args:
      ticker      : SGX ticker with .SI suffix (e.g., D05.SI)
      period      : Lookback window (90d for daily, 1y for weekly)
      interval    : Bar size ("1d" daily, "1wk" weekly)
      max_retries : Retry count on failure (default 3)
      timeout     : Per-request hard timeout in seconds (default 15)

    Returns:
      DataFrame with [Open, High, Low, Close, Volume] columns
      indexed by DatetimeIndex (timezone-aware).
      None on complete failure after all retries.
    """
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period=period, interval=interval, timeout=timeout)

            if hist is None or hist.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

            # Keep only the columns we need for indicator calculations
            hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            hist.dropna(inplace=True)
            return hist

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    [{ticker}] retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2)
                continue
            print(f"    [{ticker}] fetch failed after {max_retries} retries: {e}")
            return None

    return None


# ── Helper: version-tolerant column lookup ────────────
def find_col(df, prefix):
    """
    Find the first column whose name starts with the given prefix.
    Returns None if not found.

    Why this helper exists:
      pandas-ta has changed column naming conventions across versions.
      Examples encountered in production:
        - Bollinger:  "BBU_20_2.0" (older) → "BBU_20_2.0_2.0" (newer)
        - ATR:        "ATRr_14" vs "ATR_14" depending on version
      Prefix matching makes the pipeline robust to these changes without
      pinning to a specific pandas-ta version.
    """
    for col in df.columns:
        if col.startswith(prefix):
            return col
    return None


# ── Step 2: Calculate indicators ──────────────────────
def calculate_indicators(df):
    """
    Calculate all technical indicators using pandas-ta.

    Parameters follow industry-standard settings (see references at top
    of Task 2 markdown cell). All values are original defaults from the
    indicator creators, widely adopted in charting platforms
    (TradingView, Bloomberg, Yahoo Finance).

    Indicator families:
      - RSI (Wilder 1978)         → Momentum        (0-100 oscillator)
      - MACD (Appel 1979)         → Trend            (fast/slow EMA cross)
      - Bollinger Bands (2001)    → Volatility (relative position to MA)
      - ATR (Wilder 1978)         → Volatility (absolute daily range)
      - VWAP proxy                → Fair value       (volume-weighted mean)
      - Volume SMA                → Activity anomaly (vs 20-day average)

    Returns df with indicator columns appended in-place.
    """
    # RSI — momentum oscillator (0-100)
    df.ta.rsi(length=14, append=True)

    # MACD — trend-following momentum indicator
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Bollinger Bands — volatility bands around 20-day moving average
    df.ta.bbands(length=20, std=2, append=True)

    # ATR — average true range, measures daily volatility magnitude
    df.ta.atr(length=14, append=True)

    # VWAP proxy — volume weighted average price
    # Note: True VWAP is intraday only. For daily data, use cumulative
    # typical-price × volume / cumulative volume as a proxy. This gives
    # a "session-anchored" fair value estimate.
    df["VWAP_proxy"] = (
        (df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]
    ).cumsum() / df["Volume"].cumsum()

    # Volume SMA — 20-day average volume (matches Bollinger period)
    # Used to flag anomalous days (>1.5× average = elevated activity)
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()

    return df


# ── Step 3: Event detection ───────────────────────────
def detect_signals(df):
    """
    Apply rule-based logic to flag actionable technical events.

    Rules (thresholds are configurable via solution.json):
      RSI:
        > 70    → overbought (potential reversal down)
        < 30    → oversold   (potential reversal up)
        else    → neutral
      MACD (crossover-based, uses previous+latest day):
        MACD crosses above signal → bullish_crossover
        MACD crosses below signal → bearish_crossover
        else                      → neutral
      Bollinger Bands (position-based):
        Close > upper band → above_upper (bullish breakout)
        Close < lower band → below_lower (bearish breakdown)
        else               → within
      VWAP:
        Close > VWAP proxy → above (bullish bias)
        Close < VWAP proxy → below (bearish bias)
      Volume:
        > 1.5× 20-day SMA  → elevated (amplifies score)
        < 0.5× 20-day SMA  → low
        else               → normal

    NaN handling:
      First 20 days of any indicator are NaN (rolling warmup).
      All lookups defensively fall back to neutral / current close /
      current volume to prevent KeyError or NaN propagation.

    Returns dict of signal values for the most recent day.
    """
    latest = df.iloc[-1]   # Most recent trading day
    prev   = df.iloc[-2]   # Previous trading day (for crossover detection)

    signals = {}

    # ── RSI signal ────────────────────────────────────
    rsi = latest.get("RSI_14", 50)
    if pd.isna(rsi):
        rsi = 50
    if rsi > 70:
        signals["rsi_signal"] = "overbought"
    elif rsi < 30:
        signals["rsi_signal"] = "oversold"
    else:
        signals["rsi_signal"] = "neutral"
    signals["rsi"] = round(float(rsi), 2)

    # ── MACD signal (crossover detection uses prev + latest) ──
    macd_col   = find_col(df, "MACD_12_26_9")
    signal_col = find_col(df, "MACDs_12_26_9")

    if macd_col and signal_col:
        macd_now  = latest[macd_col]
        macd_prev = prev[macd_col]
        sig_now   = latest[signal_col]
        sig_prev  = prev[signal_col]

        if (pd.notna(macd_now) and pd.notna(macd_prev) and
            pd.notna(sig_now) and pd.notna(sig_prev)):
            if macd_prev < sig_prev and macd_now > sig_now:
                signals["macd_signal"] = "bullish_crossover"
            elif macd_prev > sig_prev and macd_now < sig_now:
                signals["macd_signal"] = "bearish_crossover"
            else:
                signals["macd_signal"] = "neutral"
            signals["macd_value"] = round(float(macd_now), 4)
        else:
            signals["macd_signal"] = "neutral"
            signals["macd_value"]  = 0.0
    else:
        signals["macd_signal"] = "neutral"
        signals["macd_value"]  = 0.0

    # ── Bollinger Bands signal ────────────────────────
    # Version-tolerant: pandas-ta uses both "BBU_20_2.0" (older) and
    # "BBU_20_2.0_2.0" (newer). Prefix-match handles both without
    # requiring a specific pandas-ta version pin.
    upper_col = find_col(df, "BBU_20_2.0")
    lower_col = find_col(df, "BBL_20_2.0")

    if upper_col and lower_col:
        close = latest["Close"]
        upper = latest[upper_col]
        lower = latest[lower_col]

        if pd.notna(upper) and pd.notna(lower):
            if close > upper:
                signals["bb_position"] = "above_upper"
            elif close < lower:
                signals["bb_position"] = "below_lower"
            else:
                signals["bb_position"] = "within"
        else:
            signals["bb_position"] = "within"
    else:
        signals["bb_position"] = "within"

    # ── VWAP signal ───────────────────────────────────
    vwap  = latest.get("VWAP_proxy", latest["Close"])
    close = latest["Close"]
    if pd.isna(vwap):
        vwap = close
    signals["vwap_position"] = "above" if close > vwap else "below"

    # ── ATR (volatility/risk measure) ─────────────────
    atr_col = find_col(df, "ATRr_14")
    atr_val = latest.get(atr_col, 0) if atr_col else 0
    if pd.isna(atr_val):
        atr_val = 0
    signals["atr"] = round(float(atr_val), 4)

    # ── Volume signal ─────────────────────────────────
    vol     = latest["Volume"]
    vol_sma = latest.get("Volume_SMA_20", vol)
    if pd.isna(vol_sma):
        vol_sma = vol
    if vol > vol_sma * 1.5:
        signals["volume_signal"] = "elevated"
    elif vol < vol_sma * 0.5:
        signals["volume_signal"] = "low"
    else:
        signals["volume_signal"] = "normal"

    return signals


# ── Step 4: Multi-timeframe confirmation ──────────────
def get_weekly_signal(ticker):
    """
    Fetch weekly OHLCV and compute RSI on weekly timeframe.

    Rationale:
      A signal confirmed on both daily AND weekly charts is more reliable
      than a daily-only signal. Weekly RSI filters out short-term noise
      and captures the dominant trend regime.

    Thresholds (looser than daily 70/30 because weekly bars are slower):
      Weekly RSI > 60 → bullish trend regime
      Weekly RSI < 40 → bearish trend regime
      else            → neutral

    Failure handling:
      Any fetch or computation failure → returns 'neutral' (safe default).
      This means multi-timeframe simply doesn't contribute to score,
      rather than crashing the pipeline.

    Returns: 'bullish' | 'bearish' | 'neutral'
    """
    try:
        df_weekly = fetch_ohlcv(ticker, period="1y", interval="1wk")
        if df_weekly is None or len(df_weekly) < 20:
            return "neutral"

        df_weekly.ta.rsi(length=14, append=True)
        weekly_rsi = df_weekly["RSI_14"].iloc[-1]

        if pd.isna(weekly_rsi):
            return "neutral"

        if weekly_rsi > 60:
            return "bullish"
        elif weekly_rsi < 40:
            return "bearish"
        else:
            return "neutral"
    except Exception:
        return "neutral"


# ── Step 5: Aggregate to overall signal ───────────────
def aggregate_technical_signal(signals, weekly_signal):
    """
    Combine all indicator signals into one overall direction via
    weighted scoring.

    Scoring weights (empirically tuned for SGX blue-chip characteristics):
      RSI:                bullish +1, bearish -1
      MACD (crossover):   bullish +2, bearish -2   (higher weight — strongest signal)
      Bollinger:          bullish +1, bearish -1
      VWAP:               bullish +1, bearish -1
      Weekly confirmation: bullish +1, bearish -1  (multi-timeframe bonus)
      Volume elevated:    score × 1.5              (amplification)

    Thresholds:
      score >= +2  → bullish
      score <= -2  → bearish
      else         → neutral

    Rationale for MACD +2:
      MACD crossovers are relatively rare (few per month per stock) but
      historically the highest-conviction signal. Overweight vs single-day
      indicators like RSI/Bollinger position.

    Rationale for volume amplifier (×1.5):
      High-volume days validate directional moves. If underlying score is
      +2 with elevated volume → final +3 (strong bullish). If score is 0
      → still 0 (volume alone doesn't create direction).
    """
    score = 0

    # RSI contribution
    if signals["rsi_signal"] == "oversold":
        score += 1    # Oversold = potential bounce up
    elif signals["rsi_signal"] == "overbought":
        score -= 1    # Overbought = potential pullback

    # MACD contribution (highest weight)
    if signals["macd_signal"] == "bullish_crossover":
        score += 2    # Strong bullish signal
    elif signals["macd_signal"] == "bearish_crossover":
        score -= 2    # Strong bearish signal

    # Bollinger Bands contribution
    if signals["bb_position"] == "above_upper":
        score += 1    # Bullish momentum breakout
    elif signals["bb_position"] == "below_lower":
        score -= 1    # Bearish breakdown

    # VWAP contribution
    if signals["vwap_position"] == "above":
        score += 1
    else:
        score -= 1

    # Multi-timeframe confirmation bonus
    if weekly_signal == "bullish":
        score += 1
    elif weekly_signal == "bearish":
        score -= 1

    # Volume amplification (multiplies existing score, doesn't create direction)
    if signals["volume_signal"] == "elevated":
        score = score * 1.5   # High volume strengthens existing signal

    if score >= 2:
        return "bullish"
    elif score <= -2:
        return "bearish"
    else:
        return "neutral"


# ── Step 6: Store to database (⭐ Supabase primary + SQLite fallback) ──
def store_technical_signal(ticker, date, signals, overall, weekly_signal):
    """
    Save computed technical signals to technical_signals table.
    One row per ticker per day. UNIQUE constraint prevents duplicates.

    Multi-timeframe field:
      weekly_signal — bullish/neutral/bearish from get_weekly_signal()
                     Used to confirm or contradict daily signal direction.

    Storage strategy (⭐ CHANGED from dual-write to primary + fallback):
      PRIMARY:  Supabase Postgres via mirror_to_supabase()
                Uses ON CONFLICT DO NOTHING (safe for re-runs).
      FALLBACK: SQLite (only executed if Supabase fails)
                INSERT OR IGNORE (SQLite equivalent of ON CONFLICT).

      Rationale for primary + fallback (not dual-write):
        - Dual-write duplicates all writes, which is wasteful.
        - Supabase is authoritative source of truth.
        - SQLite fallback only writes on Supabase outage (rare).
        - Keeps SQLite as disaster recovery backup only.
    """
    row = {
        "ticker":         ticker,
        "date":           date,
        "rsi":            signals["rsi"],
        "rsi_signal":     signals["rsi_signal"],
        "macd_value":     signals["macd_value"],
        "macd_signal":    signals["macd_signal"],
        "bb_position":    signals["bb_position"],
        "vwap_position":  signals["vwap_position"],
        "atr":            signals["atr"],
        "volume_signal":  signals["volume_signal"],
        "overall_signal": overall,
        "weekly_signal":  weekly_signal,
    }

    # ── PRIMARY: Supabase ────────────────────────────
    try:
        mirror_to_supabase("technical_signals", row)
        return
    except Exception as e:
        print(f"    ⚠️  Supabase failed for {ticker}: {type(e).__name__}: {str(e)[:80]}")
        print(f"       Falling back to SQLite...")

    # ── FALLBACK: SQLite ─────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO technical_signals
                (ticker, date, rsi, rsi_signal, macd_value, macd_signal,
                 bb_position, vwap_position, atr, volume_signal,
                 overall_signal, weekly_signal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, date,
            signals["rsi"],          signals["rsi_signal"],
            signals["macd_value"],   signals["macd_signal"],
            signals["bb_position"],  signals["vwap_position"],
            signals["atr"],          signals["volume_signal"],
            overall, weekly_signal
        ))
        conn.commit()
        conn.close()
        print(f"       ⚠️  Written to SQLite fallback")
    except Exception as e:
        print(f"    ❌ SQLite ALSO failed for {ticker}: {e}")


# ── Guard helper: check if today already processed ────
def check_technical_processed(date):
    """
    Check if technical_signals for given date already exist.
    Queries Supabase primary, falls back to SQLite.
    Returns count.
    """
    try:
        with SUPABASE_ENGINE.connect() as conn:
            return conn.execute(text("""
                SELECT COUNT(*) FROM technical_signals WHERE date = :d
            """), {"d": date}).scalar()
    except Exception:
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT COUNT(*) FROM technical_signals WHERE date=?", (date,)
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return 0


# ── Main runner ───────────────────────────────────────
def run_task2(stocks, date=None):
    """
    Run complete Task 2 pipeline for all stocks.

    No external model needed — pure mathematical calculation.
    Faster than Task 1 (no FinBERT), typically 30-60 seconds for 15 stocks.

    Guards:
      1. Skips entire run if today's data already exists (idempotent)
      2. Per-stock try/except: single failure doesn't abort other stocks
      3. Tracks failed tickers list for post-run investigation

    Args:
      stocks : dict mapping ticker → (name, sector), e.g. SGX_STOCKS
      date   : YYYY-MM-DD, defaults to today

    Returns:
      List of result dicts (one per successfully processed stock).
      Empty list if skipped due to existing data.
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    # ⭐ CHANGED: check via Supabase (with SQLite fallback)
    count = check_technical_processed(date)
    if count > 0:
        print(f"⚠️  Technical signals for {date} already exist. Skipping.")
        return []

    print(f"\n[Task 2] Technical Analysis — {date}")
    print("=" * 60)

    results = []
    failed  = []   # Track failed tickers for post-run visibility

    for ticker, (name, sector) in stocks.items():
        try:
            # Steps 1-2: fetch and calculate
            df = fetch_ohlcv(ticker)
            if df is None or len(df) < 30:
                print(f"  ⚠️  {name:45s} | no/insufficient price data — skipped")
                failed.append(ticker)
                continue

            df = calculate_indicators(df)

            # Step 3: detect signals
            signals = detect_signals(df)

            # Step 4: multi-timeframe check
            weekly = get_weekly_signal(ticker)

            # Step 5: aggregate
            overall = aggregate_technical_signal(signals, weekly)

            # Step 6: store (Supabase primary + SQLite fallback)
            store_technical_signal(ticker, date, signals, overall, weekly)

            # Print summary line
            icon = "🟢" if overall == "bullish" else "🔴" if overall == "bearish" else "🟡"
            print(f"  {icon} {name:45s} | "
                  f"RSI:{signals['rsi']:5.1f} {signals['rsi_signal']:11s} | "
                  f"MACD:{signals['macd_signal']:20s} | "
                  f"{overall}")

            results.append({
                "ticker":  ticker,
                "name":    name,
                "rsi":     signals["rsi"],
                "macd":    signals["macd_signal"],
                "overall": overall,
            })

        except Exception as e:
            print(f"  ❌ {name:45s} | {type(e).__name__}: {e}")
            failed.append(ticker)
            continue

    print(f"\n✅ Task 2 complete. {len(results)} stocks processed, {len(failed)} failed.")
    if failed:
        print(f"   Failed tickers: {failed}")
    return results


print("✅ Task 2 code loaded.")
print("   Storage: Supabase primary + SQLite fallback")
print("   To run: run_task2(SGX_STOCKS)")
