# ═══════════════════════════════════════════════════════════════════
# TASK 3 — Macroeconomic Signal Processor
#
# Purpose:
#   Track Singapore + global macro indicators daily and classify the
#   current macro regime. Provides "domain knowledge injection" context
#   for downstream LLM analysis (Task 5) so it can weight signals
#   appropriately given the macro backdrop.
#
# Indicators tracked (5 sources, 4 fetched daily):
#   1. SORA Interest Rate         (MAS API → fallback 1.24%)
#   2. CPI Year-on-Year           (data.gov.sg, monthly release)
#   3. USD/SGD Exchange Rate      (Yahoo Finance, with timeout)
#   4. VIX Volatility Index       (Yahoo Finance ^VIX, with timeout)
#   5. US Treasury 10Y-3M Spread  (Yahoo Finance ^TNX/^IRX, proxy for
#                                  SGS yield curve — SGS correlates
#                                  >0.85 with US Treasuries)
#
# Note on SORA:
#   MAS deprecated their public statistics API in 2025. The get_sora_rate()
#   function still tries the old endpoint (may return in future), but
#   currently falls back to manually-maintained constant 1.24% (Jun 2026).
#   Source: https://eservices.mas.gov.sg/statistics/dir/domesticinterestrates.aspx
#   Update fallback quarterly by checking the MAS website.
#
# Why US Treasury as yield curve proxy:
#   - SGS yields correlate >0.85 with US Treasury yields (empirical
#     evidence from 2014-2024 daily data)
#   - MAS doesn't provide a free API for SGS yield curve
#   - US 10Y-3M inversion is the gold-standard recession indicator
#     globally (NY Fed model, Estrella & Mishkin 1996)
#
# Regime classification (rule-based, priority order):
#   1. Recessionary:                  Yield curve inverted (< -0.2%)
#   2. Risk-Off:                      VIX > 30
#   3. High Rate / Elevated Inflation: SORA > 3% + CPI > 2.5%
#   4. Low Rate / Low Inflation:       SORA < 2% + CPI < 1.5%
#   5. Risk-On:                        VIX < 15
#   6. Neutral:                        else
#
# Storage (⭐ CHANGED — Supabase primary):
#   PRIMARY:  Supabase macro_indicators table via mirror_to_supabase()
#   FALLBACK: SQLite (only if Supabase fails)
#
# Resilience notes (added after 6/29 production incident):
#   - All yfinance calls use timeout=15s to prevent indefinite hangs
#     (Yahoo's 2026 anti-scraping changes caused Task 2/3/4 hangs)
#   - Each indicator fails independently → pipeline gracefully degrades
#     (e.g. if VIX fetch fails, other indicators still stored, regime
#     falls back to "Neutral")
#   - period="5d" (not "1d") for FX/VIX/yield fetches — protects against
#     weekend/holiday empty responses since 5 days always contains at
#     least 1 trading day
# ═══════════════════════════════════════════════════════════════════

import requests
import yfinance as yf
import sqlite3
from datetime import datetime
from sqlalchemy import text


# ── Step 1: SORA Interest Rate ────────────────────────
def get_sora_rate():
    """
    Fetch Singapore SORA rate from MAS API with fallback.

    SORA (Singapore Overnight Rate Average) is the benchmark interest
    rate that replaced SIBOR in 2024. Set by MAS monetary policy stance.

    MAS deprecated their public API in 2025 (returned JSON parse errors
    since mid-2025). Falls back to manually updated value (currently
    1.24% as of Jun 2026).
    Source: https://eservices.mas.gov.sg/statistics/dir/domesticinterestrates.aspx

    Returns:
      (rate, source) tuple where source is "MAS API" or "fallback"
      for audit trail in data_source column.
    """
    try:
        url = "https://eservices.mas.gov.sg/statistics/api/v1/bondsandbills/m/sora"
        r   = requests.get(url, params={"rows": 5, "sort": "end_of_day desc"},
                           timeout=10)
        records = r.json().get("result", {}).get("records", [])
        if records:
            rate = float(records[0].get("comp_sora_1m", 1.24))
            print(f"  SORA (MAS API)        : {rate}%")
            return rate, "MAS API"
    except Exception as e:
        print(f"  SORA API failed       : {e}")
    # Updated fallback to current SORA level (Jun 2026: ~1.24%)
    print("  SORA (fallback)       : 1.24%")
    return 1.24, "fallback"


# ── Step 2: CPI Year-on-Year ──────────────────────────
def get_cpi_yoy():
    """
    Fetch Singapore CPI YoY change from data.gov.sg.

    CPI (Consumer Price Index) All-Items YoY tells us inflation pressure.
    MAS's informal comfort zone is ~2% core inflation. Above 2.5% or
    below 1.5% triggers macro regime classification changes.

    Data source: data.gov.sg (Singapore government open data portal)
    Resource: d_bdaff844e3ef89d39fceb962ff8f0791 (Consumer Price Index)
    Updated monthly by SingStat.

    Computation:
      YoY % = (current month CPI / same month 12 months ago - 1) × 100
      Uses the "All Items" DataSeries row from the CPI table.

    Returns:
      YoY percentage rounded to 2 decimals, or None on any failure.
      Regime classifier treats None as "unknown" and falls through to
      other indicators for regime determination.
    """
    try:
        url    = "https://data.gov.sg/api/action/datastore_search"
        params = {
            "resource_id": "d_bdaff844e3ef89d39fceb962ff8f0791",
            "limit": 5
        }
        r      = requests.get(url, params=params, timeout=10)
        data   = r.json()
        fields = data["result"]["fields"]
        month_cols = [f["id"] for f in fields if f["id"] != "DataSeries"]
        all_items  = next(
            (rec for rec in data["result"]["records"]
             if rec.get("DataSeries") == "All Items"), None
        )
        if all_items and len(month_cols) >= 13:
            latest    = float(all_items[month_cols[0]])
            last_year = float(all_items[month_cols[12]])
            yoy       = round((latest / last_year - 1) * 100, 2)
            print(f"  CPI YoY               : {yoy}%")
            return yoy
    except Exception as e:
        print(f"  CPI fetch failed      : {e}")
    return None


# ── Step 3: USD/SGD Exchange Rate ─────────────────────
def get_usd_sgd():
    """
    Fetch USD/SGD exchange rate from Yahoo Finance with timeout.

    Yahoo Finance quotes SGD/USD (SGD per 1 USD, inverted). We convert
    to USD/SGD (USD per 1 SGD) for intuitive interpretation:
      - Rate up   → SGD strengthening vs USD (bullish for SGX)
      - Rate down → SGD weakening vs USD (bearish for exporters)

    Uses period="5d" instead of "1d" so weekend/holiday runs still get
    valid data (5-day window always includes ≥1 trading day).

    Timeout=15s prevents indefinite hangs (Yahoo's 2026 anti-scraping).

    Returns: rate rounded to 4 decimals, or None on failure.
    """
    try:
        fx   = yf.Ticker("SGDUSD=X")
        hist = fx.history(period="5d", timeout=15)   # timeout + 5d window
        if not hist.empty:
            rate = round(1 / float(hist["Close"].iloc[-1]), 4)
            print(f"  USD/SGD               : {rate}")
            return rate
    except Exception as e:
        print(f"  FX fetch failed       : {e}")
    return None


# ── Step 4: VIX (Fear Index) ──────────────────────────
def get_vix():
    """
    Fetch CBOE VIX volatility index with timeout.

    VIX measures expected 30-day S&P 500 volatility implied by options
    prices. Widely used as global "fear gauge". Correlates strongly
    with SGX during risk-off episodes (contagion via foreign flows).

    Interpretation (Whaley 2009, CBOE convention):
      < 15  : Risk-On     (market complacency, low fear)
      15-20 : Calm
      20-30 : Moderate uncertainty
      > 30  : Risk-Off    (market stress; 2008 GFC, 2020 COVID both >30)

    Yahoo Finance ticker: ^VIX
    Uses period="5d" + timeout=15s (see get_usd_sgd for rationale).

    Returns: VIX level rounded to 2 decimals, or None on failure.
    """
    try:
        vix  = yf.Ticker("^VIX")
        hist = vix.history(period="5d", timeout=15)   # timeout
        if not hist.empty:
            value = round(float(hist["Close"].iloc[-1]), 2)
            print(f"  VIX (Fear Index)      : {value}")
            return value
    except Exception as e:
        print(f"  VIX fetch failed      : {e}")
    return None


# ── Step 5: US Treasury Yield Spread ──────────────────
def get_us_yield_spread():
    """
    Calculate US Treasury 10Y - 3M spread. Proxy for SGS yield curve.

    Why US Treasuries (not SGS):
      - SGS yields correlate >0.85 with US Treasuries (empirical
        finding from 2014-2024 data)
      - MAS doesn't provide a free API for SGS yield curve
      - US 10Y-3M is the gold-standard global recession indicator
        (NY Fed research, Estrella & Mishkin 1996)

    Interpretation:
      Spread > 1.0%    : Normal, healthy economy
      0% to 1.0%       : Flattening curve, caution
      < 0% (inverted)  : Classic recession indicator
                         Historically preceded recessions by 6-18 months
                         (NY Fed model precision ~85%)
      < -0.2%          : Strongly inverted, classified as Recessionary regime

    Yahoo Finance tickers:
      ^TNX = 10Y Treasury yield (%)
      ^IRX = 3-month Treasury yield (%)

    Defensive handling (added after 6/29 debugging):
      1. dropna() removes NaN values from Close series (Yahoo sometimes
         returns partial data for holidays)
      2. Re-check after cleaning — if all NaN, return None
      3. Sanity check: yields must be > 0 (US yields have never been
         negative historically; negative value indicates data error)
      4. timeout=15s on both fetches
      5. Any failure returns None → regime classifier gracefully skips
         this indicator rather than crashing

    Returns:
      Spread rounded to 2 decimals (e.g. 0.71 means +0.71%), or None
      on any failure.
    """
    try:
        # ^TNX = 10Y Treasury yield (%)
        # ^IRX = 3-month Treasury yield (%)
        t10 = yf.Ticker("^TNX").history(period="5d", timeout=15)   # timeout
        t3m = yf.Ticker("^IRX").history(period="5d", timeout=15)   # timeout

        if t10.empty or t3m.empty:
            print(f"  Yield spread fail     : empty data from yfinance")
            return None

        # Drop NaN values from Close column (holidays / partial data)
        t10_clean = t10["Close"].dropna()
        t3m_clean = t3m["Close"].dropna()

        # Re-check after cleaning
        if len(t10_clean) == 0 or len(t3m_clean) == 0:
            print(f"  Yield spread fail     : all values were NaN after cleaning")
            return None

        # Take most recent valid value
        y10 = float(t10_clean.iloc[-1])
        y3m = float(t3m_clean.iloc[-1])

        # Sanity check: yields must be positive
        # (US Treasury yields have never been negative historically)
        if y10 <= 0 or y3m <= 0:
            print(f"  Yield spread fail     : invalid yields y10={y10}, y3m={y3m}")
            return None

        spread = round(y10 - y3m, 2)
        print(f"  US 10Y-3M Spread      : {spread}% (10Y={y10:.2f}%, 3M={y3m:.2f}%)")
        return spread

    except Exception as e:
        print(f"  Yield spread fail     : {e}")
        return None


# ── Regime classifier ─────────────────────────────────
def classify_regime(sora, cpi, vix, yield_spread):
    """
    Rule-based regime classification using 5 indicators.

    Priority order (most extreme conditions first):
      1. Recessionary:                  Yield curve inverted (< -0.2%)
                                        Overrides everything else
      2. Risk-Off:                      VIX > 30 (market panic)
      3. High Rate / Elevated Inflation: SORA > 3% AND CPI > 2.5%
      4. Low Rate / Low Inflation:       SORA < 2% AND CPI < 1.5%
      5. Risk-On:                       VIX < 15 (complacency)
      6. Neutral:                       none of above

    Design principles:
      - Priority ordering (not scoring) so extreme conditions dominate
      - AND-based rules for macro regimes (both rate AND inflation must
        agree) to avoid false positives from single-indicator noise
      - None values handled gracefully: sora=None → "Neutral" default

    Downstream use:
      Task 5 (LLM analysis) uses this regime to weight signals:
        Recessionary → down-weight bullish news, boost defensive stocks
        Risk-Off     → similar to above but less severe
        High Rate    → boost bank earnings positive news
        Low Rate     → boost REIT dividend news
    """
    # Priority 1: recession warning (highest priority)
    if yield_spread is not None and yield_spread < -0.2:
        return "Recessionary"

    # Priority 2: market fear
    if vix and vix > 30:
        return "Risk-Off"

    # Priority 3: macro regimes (need both rate AND inflation signals)
    if sora is None:
        return "Neutral"
    if sora > 3.0 and (cpi or 0) > 2.5:
        return "High Rate / Elevated Inflation"
    if sora < 2.0 and (cpi or 0) < 1.5:
        return "Low Rate / Low Inflation"

    # Priority 4: optimistic markets (only if none of above)
    if vix and vix < 15:
        return "Risk-On"

    return "Neutral"


# ── Sector impact summary ─────────────────────────────
def get_sector_impact(regime, sora, vix, yield_spread):
    """
    Plain-English sector impact narrative for downstream LLM.

    This is 'domain knowledge injection' — pre-computed text that
    tells the LLM (Task 5) how the current macro regime affects
    different SGX sectors. Without this, the LLM would have to infer
    sector implications from raw numbers.

    Example: given regime="High Rate", the LLM reading this text
    knows that DBS/OCBC/UOB should be treated as beneficiaries
    (expanding NIMs) while Ascendas REIT / Mapletree REITs face
    valuation compression.

    Returns: Plain-English string describing sector implications.
    """
    if regime == "Recessionary":
        return (f"Yield curve inverted ({yield_spread}%). Historical recession "
                "warning. All risk assets vulnerable. Defensive sectors "
                "(REITs, Healthcare, Staples) preferred over banks and industrials.")
    elif regime == "Risk-Off":
        return (f"Market fear elevated (VIX={vix}). Investors rotating to "
                "safe havens. Banks and Industrials face headwinds. "
                "Defensive REITs and dividend stocks may outperform.")
    elif regime == "Risk-On":
        return (f"Market complacency (VIX={vix}). Risk appetite high, "
                "growth and momentum stocks favored. Watch for "
                "potential reversals if VIX spikes.")
    elif regime == "High Rate / Elevated Inflation":
        return ("High SORA rate expands bank net interest margins "
                "(bullish for banking). Elevated borrowing costs compress "
                "REIT valuations (bearish for REITs). Consumer purchasing "
                "power reduced.")
    elif regime == "Low Rate / Low Inflation":
        return ("Low SORA rate compresses bank margins (bearish for banking). "
                "Lower borrowing costs support REIT valuations (bullish for REITs). "
                "Consumer spending environment supportive.")
    else:
        return (f"Neutral macro environment. SORA={sora}%, VIX={vix}, "
                f"Yield Spread={yield_spread}%. No strong sector rotation signal.")


# ── Store to database (⭐ Supabase primary + SQLite fallback) ──
def store_macro(date, sora, cpi, usd_sgd, vix, yield_spread,
                regime, source, sector_impact):
    """
    Save all macro indicators to macro_indicators table.

    Args:
      date          : YYYY-MM-DD
      sora          : SORA rate % (or None)
      cpi           : CPI YoY %  (or None)
      usd_sgd       : USD/SGD FX rate (or None)
      vix           : VIX index level (or None)
      yield_spread  : US 10Y-3M spread % (or None)
      regime        : One of 6 regime strings from classify_regime()
      source        : "MAS API" | "fallback"  (audit trail for SORA)
      sector_impact : Plain-English narrative for LLM consumption

    Storage strategy (⭐ CHANGED from dual-write to primary + fallback):
      PRIMARY:  Supabase Postgres via mirror_to_supabase()
                Uses ON CONFLICT DO NOTHING (safe for re-runs).
      FALLBACK: SQLite (only executed if Supabase fails)
                INSERT OR IGNORE (SQLite equivalent of ON CONFLICT).
    """
    row = {
        "date":                 date,
        "sg_interest_rate_pct": sora,
        "sg_cpi_yoy_pct":       cpi,
        "usd_sgd_rate":         usd_sgd,
        "vix":                  vix,
        "us_yield_spread":      yield_spread,
        "macro_regime":         regime,
        "data_source":          source,
        "sector_impact":        sector_impact,
    }

    # ── PRIMARY: Supabase ────────────────────────────
    try:
        mirror_to_supabase("macro_indicators", row)
        return
    except Exception as e:
        print(f"    ⚠️  Supabase failed for macro {date}: {type(e).__name__}: {str(e)[:80]}")
        print(f"       Falling back to SQLite...")

    # ── FALLBACK: SQLite ─────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO macro_indicators
                (date, sg_interest_rate_pct, sg_cpi_yoy_pct, usd_sgd_rate,
                 vix, us_yield_spread, macro_regime, data_source, sector_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, sora, cpi, usd_sgd, vix, yield_spread,
              regime, source, sector_impact))
        conn.commit()
        conn.close()
        print(f"       ⚠️  Written to SQLite fallback")
    except Exception as e:
        print(f"    ❌ SQLite ALSO failed: {e}")


# ── Guard helper: check if today already processed ────
def check_macro_processed(date):
    """
    Check if macro_indicators for given date already exist.
    Queries Supabase primary, falls back to SQLite.
    Returns count.
    """
    try:
        with SUPABASE_ENGINE.connect() as conn:
            return conn.execute(text("""
                SELECT COUNT(*) FROM macro_indicators WHERE date = :d
            """), {"d": date}).scalar()
    except Exception:
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT COUNT(*) FROM macro_indicators WHERE date=?", (date,)
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return 0


# ── Main runner ───────────────────────────────────────
def run_task3(date=None):
    """
    Run complete Task 3 pipeline.

    Workflow:
      1. Skip if today already has data (idempotent)
      2. Fetch all 5 indicators in sequence
      3. Classify macro regime using priority rules
      4. Generate sector impact narrative
      5. Store to macro_indicators table (Supabase primary + SQLite fallback)

    Args:
      date : YYYY-MM-DD, defaults to today.

    Returns:
      dict of all fetched values + regime + sector_impact.
      None if skipped due to existing data.
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    # ⭐ CHANGED: check via Supabase (with SQLite fallback)
    count = check_macro_processed(date)
    if count > 0:
        print(f"⚠️  Macro indicators for {date} already exist. Skipping.")
        return None

    print(f"\n[Task 3] Macro Analysis — {date}")
    print("=" * 65)

    # ── Fetch all indicators (independent failure isolation) ──
    sora, source = get_sora_rate()
    cpi          = get_cpi_yoy()
    usd_sgd      = get_usd_sgd()
    vix          = get_vix()
    yield_spread = get_us_yield_spread()

    # ── Classify regime and generate LLM context ──
    regime = classify_regime(sora, cpi, vix, yield_spread)
    print(f"\n  Macro regime          : {regime}")

    sector_impact = get_sector_impact(regime, sora, vix, yield_spread)
    print(f"  Sector impact         : {sector_impact[:80]}...")

    # ── Store (Supabase primary + SQLite fallback) ──
    store_macro(date, sora, cpi, usd_sgd, vix, yield_spread, regime, source, sector_impact)

    result = {
        "date":                 date,
        "sg_interest_rate_pct": sora,
        "sg_cpi_yoy_pct":       cpi,
        "usd_sgd_rate":         usd_sgd,
        "vix":                  vix,
        "us_yield_spread":      yield_spread,
        "macro_regime":         regime,
        "sector_impact":        sector_impact,
    }

    print(f"\n✅ Task 3 complete.")
    print(f"   SORA            : {sora}%")
    print(f"   CPI YoY         : {cpi}%")
    print(f"   USD/SGD         : {usd_sgd}")
    print(f"   VIX             : {vix}")
    print(f"   US Yield Spread : {yield_spread}%")
    print(f"   Regime          : {regime}")
    return result


print("✅ Task 3 code loaded (SORA + CPI + USD/SGD + VIX + US Yield Spread).")
print("   Storage: Supabase primary + SQLite fallback")