# ═══════════════════════════════════════════════════════════════════
# TASK 4 — Options Flow & Market Microstructure Analysis
#
# Purpose:
#   Analyze options market positioning (Put/Call ratios, Open Interest)
#   to detect institutional "smart money" sentiment for the Singapore
#   market. Provides forward-looking directional signal to complement
#   the backward-looking news (Task 1) and technical (Task 2) analyses.
#
# Data availability constraints:
#   - SGX single-stock options:  NOT available on yfinance (free tier)
#   - STI Index options:         NOT available on yfinance
#   - Bloomberg / SGX Derivatives Feed: paid, out of project scope
#
# Solution: Use ADR/ETF proxies (US-listed, options data free via yfinance)
#   - EWS (iShares MSCI Singapore ETF): Singapore market proxy
#   - SPY (SPDR S&P 500 ETF):           Global reference signal
#
# All 15 SGX stocks share the same market-level options signal
# (per-stock signal would require paid Bloomberg data).
#
# Steps:
#   1. Fetch options chain via yfinance (aggregate all expirations)
#   2. Calculate Put/Call Volume Ratio + P/C OI Ratio
#   3. Identify unusual activity (Volume > 3× Open Interest threshold)
#   4. Gamma Exposure — DEFERRED (requires Black-Scholes pricing +
#                                 delta-hedged position estimation;
#                                 outside Month 1 scope)
#   5. Classify into options_signal (bullish / neutral / bearish)
#   6. Store one row per day (market-level, not per-stock)
#
# Threshold references (see Task 4 markdown cell for full citations):
#   P/C Ratio boundaries (0.7 / 1.3):
#     - Classic CBOE convention (McMillan 2002)
#     - Neutral band widened from 1.0 because equity markets are
#       naturally long-biased (long-run avg P/C ≈ 0.85)
#   Unusual activity (Volume > 3× OI):
#     - CBOE + Pete Najarian "Options Action" convention
#     - Signals fresh directional bets (new institutional positioning)
#
# Storage (⭐ CHANGED — Supabase primary):
#   PRIMARY:  Supabase market_options table via mirror_to_supabase()
#   FALLBACK: SQLite (only if Supabase fails)
#
# Resilience notes (added after 6/29 production incident):
#   - Per-expiration fetch wrapped in try/except (single failed expiry
#     doesn't kill entire ticker aggregation)
#   - max_expirations=10 cap prevents 20+ SPY expirations from causing
#     slow runs; short-dated activity dominates anyway
#   - Weekend skip: US options market closed Sat/Sun → yfinance would
#     return stale Friday data, causing spurious unusual_activity flags
#   - Successful/failed expiry counts tracked for post-run visibility
#   - Global socket.setdefaulttimeout(30) protects the ticker.options
#     enumeration call itself (which can't take a timeout parameter)
# ═══════════════════════════════════════════════════════════════════

import yfinance as yf
import pandas as pd
import sqlite3
import time
from datetime import datetime
from sqlalchemy import text


# ── Step 1: Fetch + aggregate options chain ───────────
def get_options_aggregates(ticker, max_expirations=10):
    """
    Aggregate options data across expirations for one ticker.

    Aggregation rationale:
      Aggregating across expiries gives a fuller picture of market
      positioning than just the nearest expiry. Short-dated (weekly)
      options reflect speculation, while longer-dated reflect hedging
      and structural positioning — combining both captures total flow.

    max_expirations cap (default 10):
      Some tickers (esp. SPY) have 20+ expirations going years out.
      Aggregating all of them:
        - Slow: 20 × ~1s per option_chain fetch = 20+ seconds
        - Low value: >6 months out has negligible daily volume
      Cap at nearest 10 expirations = full coverage of active positioning
      with predictable ~10 second runtime.

    Failure isolation:
      Each expiration is fetched in try/except. If 3 of 10 fail, we
      still return aggregates from the 7 that succeeded. If ALL fail,
      return None so pipeline knows to skip.

    Args:
      ticker          : e.g. "EWS" or "SPY"
      max_expirations : Cap on expirations processed (default 10)

    Returns:
      Dict with:
        - ticker, expirations (count succeeded), failed (count failed)
        - total_call_vol, total_put_vol
        - total_call_oi,  total_put_oi
        - unusual_count (contracts with Volume > 3× OI)
      Or None if entire ticker fetch failed.
    """
    try:
        t = yf.Ticker(ticker)

        # Get expirations list (this call can hang, but global socket
        # timeout=30s protects; we can't pass timeout here directly)
        expirations = t.options

        if not expirations:
            print(f"    [{ticker}] no expirations available")
            return None

        # Limit to nearest expirations to avoid slow runs on SPY
        expirations = list(expirations)[:max_expirations]

        total_call_vol = 0
        total_put_vol  = 0
        total_call_oi  = 0
        total_put_oi   = 0
        unusual_count  = 0
        successful_exps = 0
        failed_exps     = 0

        unusual_threshold = OPTIONS_CONFIG.get("unusual_activity_multiple", 3.0)

        for exp in expirations:
            try:
                chain = t.option_chain(exp)

                # Fill NaN with 0 (yfinance returns NaN for zero-activity contracts)
                calls = chain.calls.fillna(0)
                puts  = chain.puts.fillna(0)

                # Accumulate totals across this expiration
                total_call_vol += int(calls["volume"].sum())
                total_put_vol  += int(puts["volume"].sum())
                total_call_oi  += int(calls["openInterest"].sum())
                total_put_oi   += int(puts["openInterest"].sum())

                # Step 3: count unusual activity contracts (Vol > 3× OI)
                # Iterate per-contract because we need per-strike Vol/OI comparison
                for contracts in [calls, puts]:
                    for _, row in contracts.iterrows():
                        oi  = row["openInterest"]
                        vol = row["volume"]
                        if oi > 0 and vol > unusual_threshold * oi:
                            unusual_count += 1

                successful_exps += 1

            except Exception as e:
                failed_exps += 1
                continue   # Skip failed expiry, continue with next

        if successful_exps == 0:
            print(f"    [{ticker}] all {len(expirations)} expirations failed")
            return None

        return {
            "ticker":          ticker,
            "expirations":     successful_exps,   # only count successful ones
            "failed":          failed_exps,
            "total_call_vol":  total_call_vol,
            "total_put_vol":   total_put_vol,
            "total_call_oi":   total_call_oi,
            "total_put_oi":    total_put_oi,
            "unusual_count":   unusual_count,
        }

    except Exception as e:
        print(f"  ❌ {ticker}: {type(e).__name__}: {e}")
        return None


# ── Step 2: Calculate ratios ──────────────────────────
def calculate_pc_ratios(aggregates):
    """
    Calculate Put/Call Volume + OI ratios from aggregates.

    Two ratios computed:
      P/C Volume Ratio = Total Put Volume / Total Call Volume
        → Today's flow direction. Reactive/short-term.
      P/C OI Ratio     = Total Put OI / Total Call OI
        → Cumulative positioning. Structural/longer-term.

    Both are informative for different reasons — see classify_options_signal
    for how they're combined.

    Returns: (pc_vol, pc_oi) tuple, both rounded to 2 decimals.
             Either can be None if underlying total is zero.
    """
    if not aggregates:
        return None, None

    pc_vol = (aggregates["total_put_vol"] / aggregates["total_call_vol"]
              if aggregates["total_call_vol"] > 0 else None)
    pc_oi  = (aggregates["total_put_oi"] / aggregates["total_call_oi"]
              if aggregates["total_call_oi"] > 0 else None)

    return (round(pc_vol, 2) if pc_vol else None,
            round(pc_oi, 2)  if pc_oi  else None)


# ── Step 5: Aggregate to overall signal ───────────────
def classify_options_signal(pc_vol, pc_oi):
    """
    Combine PC Volume + OI ratios into one signal.

    Thresholds (from OPTIONS_CONFIG / solution.json):
      P/C < 0.7  : calls outnumber puts by >1.4× → bullish sentiment
      P/C > 1.3  : puts outnumber calls by >1.3× → bearish sentiment
      0.7 - 1.3  : neutral / no clear directional bias

    Combination logic — VOLUME-OVERRIDE:
      Both ratios agree (both bullish or both bearish)
        → strong signal, use the agreed direction
      Volume + OI disagree
        → Volume overrides OI. Rationale:
          - Volume = TODAY's flow (fresh directional bets)
          - OI    = LEGACY positioning (may be old hedges/carries)
          - Fresh flow is more informative about current sentiment
      Volume neutral + OI directional
        → return neutral (no strong today-signal)
      Both neutral
        → neutral

    Example (real 6/22 case):
      EWS P/C Vol = 0.58 (bullish) vs P/C OI = 1.44 (bearish)
      → Volume overrides → final: bullish

    Returns: "bullish" | "neutral" | "bearish"
    """
    bullish_below = OPTIONS_CONFIG.get("pc_volume_bullish_below", 0.7)
    bearish_above = OPTIONS_CONFIG.get("pc_volume_bearish_above", 1.3)

    # Classify each activity
    def label(r):
        if r is None:
            return "neutral"
        if r < bullish_below:
            return "bullish"
        if r > bearish_above:
            return "bearish"
        return "neutral"

    vol_label = label(pc_vol)
    oi_label  = label(pc_oi)

    # Both agree → strong signal
    if vol_label == oi_label and vol_label != "neutral":
        return vol_label

    # Volume overrides OI on conflict (today's flow > legacy positions)
    if vol_label == "bullish":
        return "bullish"
    if vol_label == "bearish":
        return "bearish"

    # Volume neutral → no strong today-signal
    return "neutral"


# ── Store to database (⭐ Supabase primary + SQLite fallback) ──
def store_market_options(date, pc_vol, pc_oi, signal, unusual_count,
                          ref_pc_vol, primary_ticker, reference_ticker):
    """
    Save market-level options signal to market_options table.
    One row per day (not per ticker).

    Schema note (post 6/20 refactor):
      This replaces the old store_options_signal() function which
      stored the same data 15 times (once per SGX stock). Since
      yfinance only provides market-level options data (via EWS ETF),
      not per-stock SGX options, market-level storage (1 row/day) is
      the correct granularity. 14× storage reduction.

    Args:
      date              : YYYY-MM-DD
      pc_vol            : Put/Call volume ratio (from EWS)
      pc_oi             : Put/Call open interest ratio (from EWS)
      signal            : "bullish" / "neutral" / "bearish"
      unusual_count     : Number of contracts with Volume > 3× OI
      ref_pc_vol        : SPY P/C volume ratio (global reference)
      primary_ticker    : "EWS" (Singapore market proxy)
      reference_ticker  : "SPY" (global market reference)

    Storage strategy (⭐ CHANGED from dual-write to primary + fallback):
      PRIMARY:  Supabase Postgres via mirror_to_supabase()
                Uses ON CONFLICT DO NOTHING (safe for re-runs).
      FALLBACK: SQLite (only executed if Supabase fails)
                INSERT OR REPLACE (SQLite equivalent).
    """
    row = {
        "date":             date,
        "pc_volume_ratio":  pc_vol,
        "pc_oi_ratio":      pc_oi,
        "options_signal":   signal,
        "unusual_count":    unusual_count,
        "ref_pc_ratio":     ref_pc_vol,
        "primary_ticker":   primary_ticker,
        "reference_ticker": reference_ticker,
    }

    # ── PRIMARY: Supabase ────────────────────────────
    try:
        mirror_to_supabase("market_options", row)
        return
    except Exception as e:
        print(f"    ⚠️  Supabase failed for market_options {date}: {type(e).__name__}: {str(e)[:80]}")
        print(f"       Falling back to SQLite...")

    # ── FALLBACK: SQLite ─────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO market_options
                (date, pc_volume_ratio, pc_oi_ratio, options_signal,
                 unusual_count, ref_pc_ratio, primary_ticker, reference_ticker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, pc_vol, pc_oi, signal,
            unusual_count, ref_pc_vol, primary_ticker, reference_ticker
        ))
        conn.commit()
        conn.close()
        print(f"       ⚠️  Written to SQLite fallback")
    except Exception as e:
        print(f"    ❌ SQLite ALSO failed: {e}")


# ── Guard helper: check if today already processed ────
def check_options_processed(date):
    """
    Check if market_options for given date already exist.
    Queries Supabase primary, falls back to SQLite.
    Returns count.
    """
    try:
        with SUPABASE_ENGINE.connect() as conn:
            return conn.execute(text("""
                SELECT COUNT(*) FROM market_options WHERE date = :d
            """), {"d": date}).scalar()
    except Exception:
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT COUNT(*) FROM market_options WHERE date=?", (date,)
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return 0


# ── Main runner ───────────────────────────────────────
def run_task4(stocks, date=None):
    """
    Run Task 4 pipeline.

    Workflow:
      1. Skip if weekend (US markets closed, would return stale data)
      2. Skip if already run for this date (idempotent)
      3. Fetch EWS options (Singapore market proxy) — Layer 1
      4. Fetch SPY options (global reference) — Layer 2 (optional)
      5. Calculate P/C ratios + classify signal
      6. Store market-level signal (one row/day, Supabase primary + SQLite fallback)

    Weekend logic:
      US options markets closed Sat/Sun → yfinance returns last Friday's
      stale data. This causes spurious unusual_activity flags because
      Volume/OI ratios get distorted when data freshness mismatches
      (weekend Vol=0 but OI still shows Friday's positions).
      Skip weekends entirely to prevent stale data pollution.
      Note: SGX is closed weekends too, so no info lost.

    Two-layer signal design:
      Layer 1 (EWS) → primary signal (what SGX is doing via ADR proxy)
      Layer 2 (SPY) → reference signal (global backdrop for context)
      Currently Layer 2 is stored but not used in signal computation.
      Future work: use SPY divergence as a regime-shift indicator.

    Args:
      stocks : dict (unused currently — market-level signal, not per-stock)
      date   : YYYY-MM-DD, defaults to today

    Returns:
      dict with pc_vol_ratio, pc_oi_ratio, spy_pc_vol_ratio,
      unusual_count, options_signal. None if skipped.
    """
    if date is None:
        date_obj = datetime.today()
        date     = date_obj.strftime("%Y-%m-%d")
    else:
        date_obj = datetime.strptime(date, "%Y-%m-%d")

    # Skip weekends (Saturday=5, Sunday=6 in Python's weekday())
    if date_obj.weekday() >= 5:
        day_name = date_obj.strftime("%A")
        print(f"⚠️  {date} is a {day_name}. US options markets closed.")
        print(f"   Skipping Task 4 to avoid stale data.")
        return None

    # ⭐ CHANGED: check via Supabase (with SQLite fallback)
    count = check_options_processed(date)
    if count > 0:
        print(f"⚠️  Options signals for {date} already exist. Skipping.")
        return None

    print(f"\n[Task 4] Options Flow Analysis — {date}")
    print("=" * 65)

    primary_ticker = OPTIONS_CONFIG.get("primary_ticker", "EWS")
    ref_ticker     = OPTIONS_CONFIG.get("reference_ticker", "SPY")

    # ── Layer 1: Singapore market proxy (EWS) ──────────
    print(f"\n  Layer 1: Singapore market ({primary_ticker})")
    print("  " + "-" * 50)
    ews_agg = get_options_aggregates(primary_ticker)

    if not ews_agg:
        print(f"  ❌ Could not fetch {primary_ticker} options")
        return None

    ews_pc_vol, ews_pc_oi = calculate_pc_ratios(ews_agg)

    print(f"    Expirations used:      {ews_agg['expirations']} (failed: {ews_agg.get('failed', 0)})")
    print(f"    Total Call Volume:     {ews_agg['total_call_vol']}")
    print(f"    Total Put Volume:      {ews_agg['total_put_vol']}")
    print(f"    Total Call OI:         {ews_agg['total_call_oi']}")
    print(f"    Total Put OI:          {ews_agg['total_put_oi']}")
    print(f"    Put/Call Volume Ratio: {ews_pc_vol}")
    print(f"    Put/Call OI Ratio:     {ews_pc_oi}")
    print(f"    Unusual activity:      {ews_agg['unusual_count']} contracts")

    # ── Layer 2: Global reference (SPY) ────────────────
    print(f"\n  Layer 2: Global reference ({ref_ticker})")
    print("  " + "-" * 50)
    spy_agg = get_options_aggregates(ref_ticker)
    spy_pc_vol = None
    if spy_agg:
        spy_pc_vol, spy_pc_oi = calculate_pc_ratios(spy_agg)
        print(f"    SPY Put/Call Volume:   {spy_pc_vol}")
        print(f"    SPY Put/Call OI:       {spy_pc_oi}")
    else:
        print(f"    ⚠️  SPY reference fetch failed (continuing without it)")

    # ── Step 5: Classify signal ───────────────────────
    signal = classify_options_signal(ews_pc_vol, ews_pc_oi)
    icon   = "🟢" if signal == "bullish" else "🔴" if signal == "bearish" else "🟡"
    print(f"\n  {icon} Overall options signal: {signal}")

    # ── Store market-level signal (one row, not 15) ────
    print(f"\n  Storing market-level options signal...")
    store_market_options(
        date=date,
        pc_vol=ews_pc_vol,
        pc_oi=ews_pc_oi,
        signal=signal,
        unusual_count=ews_agg["unusual_count"],
        ref_pc_vol=spy_pc_vol,
        primary_ticker=primary_ticker,
        reference_ticker=ref_ticker
    )
    print(f"  ✅ Saved 1 row to market_options table")

    print(f"\n✅ Task 4 complete.")
    print(f"   Singapore P/C Vol:     {ews_pc_vol}")
    print(f"   Singapore P/C OI:      {ews_pc_oi}")
    print(f"   Global P/C Vol (SPY):  {spy_pc_vol}")
    print(f"   Unusual activity:      {ews_agg['unusual_count']}")
    print(f"   Signal:                {signal}")

    return {
        "date":             date,
        "pc_vol_ratio":     ews_pc_vol,
        "pc_oi_ratio":      ews_pc_oi,
        "spy_pc_vol_ratio": spy_pc_vol,
        "unusual_count":    ews_agg["unusual_count"],
        "options_signal":   signal,
    }


print("✅ Task 4 code loaded (EWS + SPY options flow).")
print("   Storage: Supabase primary + SQLite fallback")