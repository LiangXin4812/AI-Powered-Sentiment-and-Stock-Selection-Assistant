# ═══════════════════════════════════════════════════════════════════════════
# TASK 5 — DAILY RUNNER
#
# One command per morning, alongside run_task6_daily():
#
#     run_task5_daily()
#
# What it does, mirroring Task 6's daily shape:
#   1. Detect the latest date where T1 (company news) AND T2 (technicals)
#      both exist — same completeness rule the backfill uses. T3/T4 are
#      market-wide with neutral-default fallbacks in the extractor, so they
#      don't gate the run.
#   2. Skip (ticker, date) pairs already in llm_analyses — safe to re-run
#      after a crash or quota stop; it continues where it left off.
#   3. Run the full 8-step T5 pipeline per ticker, store, print one line each.
#   4. Print the day's sentiment / warning distribution.
#
# Prereqs in-session: Setup + extract_features_row cell + Part B v7.
# Quota: 15 calls ≈ 100K tokens. On the free tier this shares the daily
# budget with any backfill still running — do the daily run FIRST so today
# never falls behind, then let the backfill eat what remains.
# ═══════════════════════════════════════════════════════════════════════════

import time as _time
import traceback as _tb
from sqlalchemy import text


def _t5_latest_complete_date():
    """Latest date with both company news and technicals present."""
    with SUPABASE_ENGINE.connect() as conn:
        r = conn.execute(text("""
            SELECT MAX(n.date)
            FROM news_articles n
            WHERE n.category = 'company'
              AND EXISTS (SELECT 1 FROM technical_signals t
                          WHERE t.date = n.date)
        """)).fetchone()
    return str(r[0]) if r and r[0] else None


def _t5_existing_tickers(date):
    with SUPABASE_ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT ticker FROM llm_analyses
            WHERE analysis_date::date = CAST(:d AS DATE)
        """), {"d": date}).fetchall()
    return {r[0] for r in rows}


def _t5_run_one(ticker, date):
    """One (ticker, date) through the 8-step pipeline. Never raises."""
    try:
        context = compose_stock_context(ticker, date)
        if context is None:
            return {"status": "skip", "reason": "no context"}

        features_result = compute_feature_contributions(context)
        if features_result is None:
            return {"status": "skip", "reason": "no features"}

        shap_result    = compute_shap_explanation(ticker, date)
        contradictions = detect_contradictions(features_result, shap_result)
        prompt         = build_llm_prompt(context, features_result,
                                          contradictions, shap_result)

        llm_output = call_groq_llm(prompt)
        if llm_output is None:
            return {"status": "fail", "reason": "LLM returned None"}

        is_valid, notes = validate_llm_output(
            llm_output,
            features_result["baseline_sentiment"],
            features_result["baseline_confidence"],
        )
        store_llm_analysis(ticker, date, llm_output, features_result,
                           contradictions, shap_result, prompt, is_valid, notes)
        return {
            "status":     "ok" if is_valid else "fallback",
            "sentiment":  llm_output.get("sentiment"),
            "confidence": llm_output.get("confidence"),
            "warning":    contradictions.get("warning_level"),
        }
    except Exception as e:
        return {"status": "error",
                "reason": f"{type(e).__name__}: {e}",
                "trace":  _tb.format_exc(limit=3)}


def run_task5_daily(pause_between=6.0, stop_after_fails=3):
    """
    Analyse all 15 stocks for the latest complete data date.
    Resume-safe: already-stored tickers are skipped, so re-running after a
    quota stop finishes the remainder.
    """
    date = _t5_latest_complete_date()

    print("\n" + "=" * 70)
    print(f"TASK 5 DAILY RUN — data date {date}")
    print("=" * 70)
    print(f"  Rule layer: {RULE_LAYER_VERSION}   model: {TASK5_ARTIFACTS.get('model_version')}")

    if date is None:
        print("  🛑 No complete T1+T2 date found — run Tasks 1-4 first.")
        return

    done = _t5_existing_tickers(date)
    todo = [t for t in SGX_STOCKS if t not in done]

    print(f"  Already stored: {len(done)}   To analyse: {len(todo)}\n")
    if not todo:
        print("  ✅ All 15 already analysed for this date. Nothing to do.")
        return

    counts     = {"ok": 0, "fallback": 0, "skip": 0, "fail": 0, "error": 0}
    sentiments = {"bullish": 0, "neutral": 0, "bearish": 0}
    warnings_  = {"low": 0, "mild": 0, "high": 0}
    consecutive_fails = 0

    for i, tkr in enumerate(todo, 1):
        r = _t5_run_one(tkr, date)
        counts[r["status"]] = counts.get(r["status"], 0) + 1

        if r["status"] in ("ok", "fallback"):
            consecutive_fails = 0
            s, w = r.get("sentiment"), (r.get("warning") or "").lower()
            if s in sentiments: sentiments[s] += 1
            if w in warnings_:  warnings_[w]  += 1
            flag = "" if r["status"] == "ok" else " ⚠️FALLBACK"
            print(f"  [{i:2d}/{len(todo)}] {tkr:<10} {s:<8} "
                  f"{r.get('confidence', 0):.2f} [{w}]{flag}")
        elif r["status"] == "skip":
            print(f"  [{i:2d}/{len(todo)}] {tkr:<10} — skipped ({r['reason']})")
        else:
            consecutive_fails += 1
            print(f"  [{i:2d}/{len(todo)}] {tkr:<10} ❌ {r.get('reason','?')[:55]}")
            if consecutive_fails >= stop_after_fails:
                print(f"\n  🛑 {stop_after_fails} consecutive failures — quota "
                      f"likely exhausted. Re-run later; completed tickers "
                      f"are skipped.")
                break

        _time.sleep(pause_between)

    written = counts["ok"] + counts["fallback"]
    print(f"\n  {'─'*56}")
    print(f"  Written: {written}   valid: {counts['ok']}   "
          f"fallback: {counts['fallback']}   failed: {counts['fail'] + counts['error']}")
    if written:
        parts = [f"{k} {sentiments[k]}" for k in ("bullish", "neutral", "bearish")]
        wparts = [f"{k} {warnings_[k]}" for k in ("low", "mild", "high")]
        print(f"  Sentiment: {' / '.join(parts)}")
        print(f"  Warnings:  {' / '.join(wparts)}")
    if counts["fail"] + counts["error"] > 0:
        print(f"  ↻ Re-run run_task5_daily() later to finish the remainder.")
    print("=" * 70)


print("✅ run_task5_daily() ready")
print("   Morning routine:  run_task5_daily()  →  run_task6_daily()")
print("   (T5 first — 15 calls; if quota dies mid-run, re-run to resume)")