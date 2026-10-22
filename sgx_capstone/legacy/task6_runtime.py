"""Task 6 serving: feature inference, stored forecasts, outcome updates and monitoring.

Selected from notebook Part B v6. Automatic retraining is disabled in this
package; candidate calibration belongs in research/purged_calibration.py.
Metadata preflight rejects known prefit artifacts but does not establish date
purging or future predictive skill. Historical overwrite evidence is retained
in the research records and report incident log.
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Load artifacts ───────────────────────────────────────────────────
ARTIFACTS_PATH_T6 = globals().get("ARTIFACTS_PATH_T6", "/app/models/task6_artifacts_v1.pkl")

with open(ARTIFACTS_PATH_T6, "rb") as f:
    TASK6_ARTIFACTS = pickle.load(f)

FEATURE_NAMES_T6 = TASK6_ARTIFACTS["feature_names"]
SCALER_T6        = TASK6_ARTIFACTS["scaler"]
LABEL_ENC_T6     = TASK6_ARTIFACTS["label_encoder"]
FORWARD_HORIZON  = TASK6_ARTIFACTS["forward_horizon"]
MODEL_VERSION_T6 = TASK6_ARTIFACTS["model_version"]
RETURN_THRESHOLD_T6 = TASK6_ARTIFACTS["return_threshold"]
MAX_MISSING_T6 = globals().get("MAX_MISSING_T6", 3)

# NOTE: BEST_MODEL_NAME is set BELOW via get_current_champion() (DB-first).
# pkl's best_model_name field is only used as a first-run fallback when
# task6_champion_log has no history yet. See STEP 2 for the resolver.
PKL_FALLBACK_CHAMPION = TASK6_ARTIFACTS.get("best_model_name", "XGBoost")

AVAILABLE_MODELS = {
    "LogReg":  TASK6_ARTIFACTS["logreg_calibrated"],
    "RandomF": TASK6_ARTIFACTS["rf_calibrated"],
    "XGBoost": TASK6_ARTIFACTS["xgb_calibrated"],
}
AVAILABLE_MODELS = {k: v for k, v in AVAILABLE_MODELS.items() if v is not None}

print(f"✅ Task 6 artifacts loaded (pkl only — champion resolved from DB below)")
print(f"   Model version:    {MODEL_VERSION_T6}")
print(f"   pkl fallback:     {PKL_FALLBACK_CHAMPION}")
print(f"   Available models: {list(AVAILABLE_MODELS.keys())}")
print(f"   Features:         {len(FEATURE_NAMES_T6)}")
print(f"   Horizon:          {FORWARD_HORIZON} days")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Core prediction function (per model)
# ═══════════════════════════════════════════════════════════════════════════

def predict_task6(ticker, prediction_date, model_name=None, verbose=True):
    """Single (ticker, date, model) prediction.

    If model_name is not passed, uses the current champion resolved from DB
    (task6_champion_log), never from the potentially-stale in-memory global.
    """
    if model_name is None:
        model_name = get_current_champion()
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not available. "
                         f"Choose from {list(AVAILABLE_MODELS.keys())}")

    features_dict = extract_features_row(ticker, prediction_date)
    if features_dict is None:
        if verbose:
            print(f"  ⚠️  {ticker} @ {prediction_date}: no features available")
        return {"error": "no_features", "predicted_class": None, "probabilities": None}

    # A missing input takes its training mean (the scaler's mean_), not zero.
    # Zero is an extreme value for most of these features (RSI 0, VIX 0), so a
    # zero fill silently invents a signal. More than MAX_MISSING_T6 gaps and
    # the forecast is refused rather than built mostly from averages.
    row, missing = [], []
    for i, f in enumerate(FEATURE_NAMES_T6):
        v = features_dict.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = float(SCALER_T6.mean_[i])
            missing.append(f)
        row.append(v)
    if len(missing) > MAX_MISSING_T6:
        if verbose:
            print(f"  ⚠️  {ticker} @ {prediction_date}: {len(missing)} features "
                  f"missing ({missing}); forecast refused")
        return {"error": "too_many_missing_features", "predicted_class": None,
                "probabilities": None, "missing_features": missing}
    X_row = pd.DataFrame([row], columns=FEATURE_NAMES_T6)

    X_input = SCALER_T6.transform(X_row) if model_name == "LogReg" else X_row

    model = AVAILABLE_MODELS[model_name]
    proba = model.predict_proba(X_input)[0]
    pred_enc = int(np.argmax(proba))
    pred_cls = LABEL_ENC_T6.inverse_transform([pred_enc])[0]
    prob_dict = {cls: float(proba[i]) for i, cls in enumerate(LABEL_ENC_T6.classes_)}
    # Provisional maturity gate. The exact third SGX trading session is
    # written after prices arrive; calendar days would mature weekends early.
    forecast_dt = (datetime.strptime(prediction_date, "%Y-%m-%d") +
                   timedelta(days=FORWARD_HORIZON + 7)).strftime("%Y-%m-%d")

    return {
        "ticker":           ticker,
        "prediction_date":  prediction_date,
        "forecast_target":  forecast_dt,
        "predicted_class":  pred_cls,
        "probabilities":    prob_dict,
        "confidence":       float(np.max(proba)),
        "model_used":       model_name,
        "model_version":    MODEL_VERSION_T6,
        "feature_count":    len(FEATURE_NAMES_T6) - len(missing),
        "missing_features": missing,
    }

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Storage
# ═══════════════════════════════════════════════════════════════════════════

CREATE_T6_PRED_TABLE = """
CREATE TABLE IF NOT EXISTS task6_predictions (
    id                SERIAL PRIMARY KEY,
    ticker            VARCHAR(20)   NOT NULL,
    prediction_date   DATE          NOT NULL,
    forecast_target   DATE          NOT NULL,
    horizon_days      INTEGER       NOT NULL DEFAULT 3,
    predicted_class   VARCHAR(20)   NOT NULL,
    prob_bullish      FLOAT,
    prob_neutral      FLOAT,
    prob_bearish      FLOAT,
    confidence        FLOAT,
    actual_return_3d  FLOAT,
    actual_class      VARCHAR(20),
    is_correct        BOOLEAN,
    is_champion       BOOLEAN       DEFAULT FALSE,
    model_used        VARCHAR(50)   NOT NULL,
    model_version     VARCHAR(20)   DEFAULT 'v1',
    feature_count     INTEGER,
    created_at        TIMESTAMP     DEFAULT NOW(),
    updated_at        TIMESTAMP     DEFAULT NOW(),
    UNIQUE (ticker, prediction_date, model_used, model_version)
);
"""

# For champion history (track when champion changes)
CREATE_CHAMPION_LOG = """
CREATE TABLE IF NOT EXISTS task6_champion_log (
    id                SERIAL PRIMARY KEY,
    changed_at        TIMESTAMP     DEFAULT NOW(),
    old_champion      VARCHAR(50),
    new_champion      VARCHAR(50)   NOT NULL,
    reason            VARCHAR(200),
    metric_snapshot   JSONB
);
"""

with SUPABASE_ENGINE.begin() as conn:
    conn.execute(text(CREATE_T6_PRED_TABLE))
    conn.execute(text(CREATE_CHAMPION_LOG))
print("✅ task6_predictions + task6_champion_log tables ready")

# ═══════════════════════════════════════════════════════════════════════════
# CHAMPION STATE: DB-first source of truth
# ═══════════════════════════════════════════════════════════════════════════
# The pickle can go stale (Google Drive cache, kernel restart, manual edits),
# so we treat task6_champion_log as the authoritative source. Every predict
# and every is_champion decision queries DB. The pkl's best_model_name field
# is used only as a fallback when the log has no rows yet (first-run case).

def get_current_champion():
    """
    Return the current champion by querying task6_champion_log.
    Falls back to pkl's best_model_name only when the log is empty
    (first ever run, before any retrain_weekly has fired).
    On DB errors, falls back to pkl and prints a warning.
    """
    try:
        with SUPABASE_ENGINE.connect() as conn:
            r = conn.execute(text("""
                SELECT new_champion
                FROM task6_champion_log
                ORDER BY changed_at DESC
                LIMIT 1
            """)).fetchone()
        if r and r[0]:
            return r[0]
        # No swap history yet — use pkl's initial champion
        print(f"  ℹ️  No swap history in task6_champion_log; "
              f"using pkl fallback: {PKL_FALLBACK_CHAMPION}")
        return PKL_FALLBACK_CHAMPION
    except Exception as e:
        print(f"  ⚠️  DB query for champion failed ({e!r}); "
              f"using pkl fallback: {PKL_FALLBACK_CHAMPION}")
        return PKL_FALLBACK_CHAMPION


# Initialize runtime champion from DB (source of truth)
BEST_MODEL_NAME = get_current_champion()
print(f"✅ Champion (from DB): {BEST_MODEL_NAME}")
print(f"   Challengers:       {[m for m in AVAILABLE_MODELS if m != BEST_MODEL_NAME]}")

def store_prediction(pred, is_champion=False):
    """Upsert one prediction to Supabase."""
    if pred.get("error"):
        return False
    q = text("""
        INSERT INTO task6_predictions (
            ticker, prediction_date, forecast_target, horizon_days,
            predicted_class, prob_bullish, prob_neutral, prob_bearish,
            confidence, is_champion, model_used, model_version, feature_count
        ) VALUES (
            :ticker, :prediction_date, :forecast_target, :horizon,
            :predicted_class, :pb, :pn, :pbear,
            :confidence, :is_champion, :model_used, :model_version, :feature_count
        )
        ON CONFLICT (ticker, prediction_date, model_used, model_version)
        DO UPDATE SET
            forecast_target = EXCLUDED.forecast_target,
            predicted_class = EXCLUDED.predicted_class,
            prob_bullish    = EXCLUDED.prob_bullish,
            prob_neutral    = EXCLUDED.prob_neutral,
            prob_bearish    = EXCLUDED.prob_bearish,
            confidence      = EXCLUDED.confidence,
            is_champion     = EXCLUDED.is_champion,
            feature_count   = EXCLUDED.feature_count,
            updated_at      = NOW();
    """)
    with SUPABASE_ENGINE.begin() as conn:
        conn.execute(q, {
            "ticker":          pred["ticker"],
            "prediction_date": pred["prediction_date"],
            "forecast_target": pred["forecast_target"],
            "horizon":         FORWARD_HORIZON,
            "predicted_class": pred["predicted_class"],
            "pb":              pred["probabilities"].get("bullish", 0),
            "pn":              pred["probabilities"].get("neutral", 0),
            "pbear":           pred["probabilities"].get("bearish", 0),
            "confidence":      pred["confidence"],
            "is_champion":     is_champion,
            "model_used":      pred["model_used"],
            "model_version":   pred["model_version"],
            "feature_count":   pred["feature_count"],
        })
    return True

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Batch predict — ALL 3 models per ticker (Champion + Challengers)
# ═══════════════════════════════════════════════════════════════════════════

def predict_all_tickers(prediction_date, tickers=None, store=True):
    """
    Champion + 2 Challengers each predict for each ticker.
    Champion's prediction is marked is_champion=TRUE (used downstream by Task 7).

    The champion is resolved FROM DB at the start of this call — not from
    the in-memory BEST_MODEL_NAME — so a kernel-restart-loaded stale pkl
    can never cause the wrong model to be flagged as champion in storage.
    """
    # DB is source of truth. Never trust in-memory BEST_MODEL_NAME for storage.
    current_champion = get_current_champion()

    # Also refresh the module-level global so downstream prints/queries
    # see the same value.
    global BEST_MODEL_NAME
    if BEST_MODEL_NAME != current_champion:
        print(f"  ℹ️  Runtime champion updated: {BEST_MODEL_NAME} → {current_champion} (from DB)")
        BEST_MODEL_NAME = current_champion

    tickers = tickers or list(SGX_STOCKS.keys())
    print(f"\n{'='*70}")
    print(f"BATCH PREDICTION — {len(tickers)} tickers × {len(AVAILABLE_MODELS)} models")
    print(f"{'='*70}")
    print(f"  Data date: {prediction_date}   Champion: {current_champion} (DB)")
    print(f"  Store: {store}\n")

    champion_summary = {"bullish": 0, "neutral": 0, "bearish": 0}
    all_results      = []
    ok_count = err_count = 0

    for i, tkr in enumerate(tickers, 1):
        line_pieces = [f"  [{i:2d}/{len(tickers)}] {tkr:<10}"]
        for m in AVAILABLE_MODELS:
            pred = predict_task6(tkr, prediction_date, m, verbose=False)
            all_results.append(pred)
            if pred.get("error"):
                line_pieces.append(f"{m}:err")
                err_count += 1
                continue
            if store:
                store_prediction(pred, is_champion=(m == current_champion))
            if m == current_champion:
                champion_summary[pred["predicted_class"]] += 1
                ok_count += 1
                line_pieces.append(f"⭐{m}={pred['predicted_class']}({pred['confidence']:.2f})")
            else:
                line_pieces.append(f"{m}={pred['predicted_class']}({pred['confidence']:.2f})")
        print("  ".join(line_pieces))

    total_stored = len(tickers) * len(AVAILABLE_MODELS)
    print(f"\n  {'─'*60}")
    print(f"  ✅ Champion predictions: {ok_count}    ⚠️  Errors: {err_count}")
    print(f"  📥 Total rows stored: ~{total_stored} (3 models × {len(tickers)} tickers)")

    if ok_count > 0:
        print(f"\n  Champion ({current_champion}) sentiment distribution:")
        for cls in ["bullish", "neutral", "bearish"]:
            pct = champion_summary[cls] / ok_count * 100
            bar = "█" * int(pct / 3)
            print(f"     {cls:<8s} {champion_summary[cls]:>2d}  ({pct:>5.1f}%)  {bar}")
    print(f"  {'─'*60}")
    return all_results

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Update actual labels
# ═══════════════════════════════════════════════════════════════════════════

def update_actual_labels(prediction_date, threshold=None):
    import yfinance as yf
    threshold = threshold or RETURN_THRESHOLD_T6

    print(f"\n  Updating actuals for {prediction_date}...")
    with SUPABASE_ENGINE.connect() as conn:
        preds = conn.execute(text("""
            SELECT id, ticker, prediction_date, forecast_target,
                   predicted_class, model_used
            FROM task6_predictions
            WHERE prediction_date = :d AND actual_class IS NULL
        """), {"d": prediction_date}).fetchall()

    if not preds:
        print("  (nothing to update — either already done or none to do)")
        return

    updated = 0
    # Group by ticker to avoid duplicate yfinance calls (3 models same ticker)
    tickers_to_fetch = list(set(r[1] for r in preds))
    price_cache = {}
    for tkr in tickers_to_fetch:
        try:
            fetch_end = (datetime.strptime(prediction_date, "%Y-%m-%d") +
                         timedelta(days=FORWARD_HORIZON + 7)).strftime("%Y-%m-%d")
            df = yf.Ticker(tkr).history(start=prediction_date, end=fetch_end)
            if len(df) < FORWARD_HORIZON + 1:
                continue
            df = df.sort_index()
            start_price = df["Close"].iloc[0]
            end_price   = df["Close"].iloc[FORWARD_HORIZON]
            ret_pct = (end_price - start_price) / start_price * 100
            if ret_pct > threshold:
                actual = "bullish"
            elif ret_pct < -threshold:
                actual = "bearish"
            else:
                actual = "neutral"
            target_date = df.index[FORWARD_HORIZON].date().isoformat()
            price_cache[tkr] = (float(ret_pct), actual, target_date)
        except Exception:
            continue

    for row in preds:
        p_id, tkr, p_date, f_target, pred_cls, model = row
        if tkr not in price_cache:
            continue
        ret_pct, actual, target_date = price_cache[tkr]
        is_correct = (actual == pred_cls)
        with SUPABASE_ENGINE.begin() as conn:
            conn.execute(text("""
                UPDATE task6_predictions
                SET actual_return_3d = :r, actual_class = :c,
                    forecast_target = :target,
                    is_correct = :ok, updated_at = NOW()
                WHERE id = :id
            """), {"r": ret_pct, "c": actual, "target": target_date,
                   "ok": is_correct, "id": p_id})
        updated += 1
    print(f"  ✅ Updated {updated} rows")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Evaluate — per-model rolling F1
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_predictions(start_date=None, end_date=None):
    """Per-model F1 + Accuracy on matured predictions in date range."""
    filters = ["actual_class IS NOT NULL"]
    params  = {}
    if start_date:
        filters.append("prediction_date >= :sd"); params["sd"] = start_date
    if end_date:
        filters.append("prediction_date <= :ed"); params["ed"] = end_date
    where = " AND ".join(filters)
    q = f"""
        SELECT model_used, predicted_class, actual_class, is_correct
        FROM task6_predictions
        WHERE {where}
        ORDER BY prediction_date, ticker
    """
    with SUPABASE_ENGINE.connect() as conn:
        rows = conn.execute(text(q), params).fetchall()

    if not rows:
        print("  (no matured predictions in range)")
        return None

    df = pd.DataFrame(rows, columns=["model", "predicted", "actual", "correct"])
    print(f"\n  Total matured predictions: {len(df)}")
    print(f"  {'Model':<10} {'N':>5} {'Accuracy':>10} {'F1 (wtd)':>10} {'Role':>12}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*10} {'-'*12}")
    current_champion = get_current_champion()   # DB source of truth

    per_model = {}
    for m in AVAILABLE_MODELS:
        d = df[df["model"] == m]
        if len(d) == 0:
            per_model[m] = None
            print(f"  {m:<10} {0:>5} {'--':>10} {'--':>10} {'':>12}")
            continue
        acc = d["correct"].mean()     # average correct = accuracy
        f1  = f1_score(d["actual"], d["predicted"], average="weighted", zero_division=0)
        role = "⭐ Champion" if m == current_champion else "  Challenger"
        per_model[m] = {"n": len(d), "accuracy": acc, "f1": f1}
        print(f"  {m:<10} {len(d):>5} {acc:>10.3f} {f1:>10.3f} {role:>12}")

    return per_model

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Helpers for automation
# ═══════════════════════════════════════════════════════════════════════════

def get_latest_complete_data_date():
    with SUPABASE_ENGINE.connect() as conn:
        r = conn.execute(text("""
            SELECT MAX(n.date) FROM news_articles n
            WHERE n.category = 'company'
              AND EXISTS (SELECT 1 FROM technical_signals t
                          WHERE t.date = n.date AND t.ticker = n.ticker)
              AND EXISTS (SELECT 1 FROM macro_indicators m WHERE m.date = n.date)
              AND EXISTS (SELECT 1 FROM market_options o WHERE o.date = n.date)
              AND EXISTS (SELECT 1 FROM news_articles p
                          WHERE p.ticker='POLITICAL' AND p.date = n.date)
        """)).fetchone()
    return str(r[0]) if r[0] else None

def get_predictions_needing_actual(today_str):
    with SUPABASE_ENGINE.connect() as conn:
        r = conn.execute(text("""
            SELECT DISTINCT prediction_date FROM task6_predictions
            WHERE forecast_target < :today AND actual_class IS NULL
            ORDER BY prediction_date
        """), {"today": today_str}).fetchall()
    return [str(x[0]) for x in r]

def already_predicted(prediction_date):
    """True if we've stored predictions for this date (any model)."""
    with SUPABASE_ENGINE.connect() as conn:
        r = conn.execute(text("""
            SELECT COUNT(*) FROM task6_predictions WHERE prediction_date = :d
        """), {"d": prediction_date}).fetchone()
    return r[0] > 0

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: retrain_weekly  (v2 — out-of-fold calibration, 19 Sep)
#
# Three fixes, all found on 19 Sep:
#
# 1. CALIBRATION LEAK (the big one)
#    v1 fitted the sigmoid calibrator on the SAME rows the model was trained
#    on (cv="prefit"). A random forest fits its training data almost
#    perfectly, so the calibrator learned "the model is always right" and
#    pushed probabilities to 0/1. Live result 17 Aug-18 Sep: 111 of 255
#    forecasts claimed >= 0.9 and those were right 35% of the time.
#    v2 calibrates out-of-fold with a TimeSeriesSplit.
#
# 2. CHAMPION CHURN
#    v1 swapped champion on any F1 lead; four swaps in four weeks on windows
#    sharing 75% of their data. v2 needs CHAMPION_MARGIN.
#
# 3. STALE METADATA / SILENT PERSIST FAILURE
#    v2 records the real train window and sample count, and reads the pkl
#    back after writing.
# ═══════════════════════════════════════════════════════════════════════════

CHAMPION_MARGIN = 0.03     # challenger must beat champion F1 by this much
CALIB_SPLITS    = 4        # TimeSeriesSplit folds for out-of-fold calibration


def retrain_weekly(*args, **kwargs):
    """Serving is pinned. Build and validate a purged research candidate first."""
    raise RuntimeError(
        "Automatic/legacy retraining is disabled. Use research/purged_calibration.py "
        "with exact label maturities and future-date evaluation before promotion."
    )


def t6_preflight(path=None, verbose=True):
    """True when the production pkl holds out-of-fold calibrated models.

    Two things are checked, because either alone can be stale:
      • the 'calibration' field written by retrain_weekly v2
      • the estimators themselves (a prefit fit keeps cv='prefit')
    """
    path = path or ARTIFACTS_PATH_T6
    with open(path, "rb") as f:
        art = pickle.load(f)

    calib = str(art.get("calibration", ""))
    leaky = [n for n in ("logreg_calibrated", "rf_calibrated", "xgb_calibrated")
             if art.get(n) is not None and str(getattr(art[n], "cv", None)) == "prefit"]
    ok = calib.startswith("sigmoid, out-of-fold") and not leaky

    if verbose:
        print(f"  pkl          : {path}")
        print(f"  trained_at   : {art.get('trained_at', '—')}")
        print(f"  train_window : {art.get('train_window', '—')}  "
              f"({art.get('n_train_samples', '—')} samples)")
        print(f"  calibration  : {art.get('calibration', '— (missing: Part A artefact)')}")
        if leaky:
            print(f"  ❌ prefit (leaky) estimators: {leaky}")
        print("  " + ("✅ production models are out-of-fold — safe to forecast"
                      if ok else
                      "🛑 leaky models in production — run retrain_weekly() before forecasting"))
    return ok

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: 🎯 MAIN — run_task6_daily
# ═══════════════════════════════════════════════════════════════════════════

def run_task6_daily():
    """
    Daily runner. Automates:
      1. Detect latest complete T1-4 date
      2. Generate predictions for that date (all 3 models)
      3. Update actuals for matured predictions
      4. Print rolling accuracy per model
      5. Report that retraining is a separate reviewed research operation.
    """
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    print("\n" + "=" * 70)
    print(f"TASK 6 DAILY RUN — {today_str} ({today.strftime('%A')})")
    print("=" * 70)

    # [0/5] Refuse to forecast with leaky models (22 Sep incident).
    print("[0/5] Preflight: production model calibration")
    if not t6_preflight():
        print("\n  🛑 Stopped before predicting. Run retrain_weekly() to rebuild the")
        print("     models with out-of-fold calibration, then run this again.")
        return
    print()
    print(f"  Champion: {get_current_champion()} {MODEL_VERSION_T6}   (source: DB)\n")

    # [1/5] Detect data
    print("─" * 70)
    print("[1/5] Detecting latest complete T1-4 date...")
    latest = get_latest_complete_data_date()
    if not latest:
        print("  ❌ No complete T1-4 data. Run Tasks 1-4 first.")
        return
    print(f"  ✅ Latest complete date: {latest}")

    # [2/5] Generate predictions
    print("\n" + "─" * 70)
    print(f"[2/5] Generating predictions for {latest} (all 3 models)...")
    if already_predicted(latest):
        print(f"  ⏭️  Predictions for {latest} already exist — skipping.")
    else:
        predict_all_tickers(latest, store=True)

    # [3/5] Update actual labels
    print("\n" + "─" * 70)
    print(f"[3/5] Updating actual labels for matured predictions...")
    dates_to_update = get_predictions_needing_actual(today_str)
    if not dates_to_update:
        print("  (no matured predictions awaiting actuals)")
    else:
        print(f"  Found {len(dates_to_update)} prediction dates to update:")
        for d in dates_to_update:
            update_actual_labels(d)

    # [4/5] Rolling accuracy per model
    print("\n" + "─" * 70)
    print(f"[4/5] Rolling 14-day accuracy per model...")
    start_14d = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    evaluate_predictions(start_date=start_14d, end_date=today_str)

    print("\n  [5/5] Automatic retraining disabled; serving artifact remains pinned.")

    print("\n" + "=" * 70)
    print(f"✅ Done. Next daily run tomorrow ({(today + timedelta(days=1)).strftime('%Y-%m-%d %A')}).")
    print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# HOW TO USE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("✅ TASK 6 PART B v6 ready — DB-first champion + out-of-fold retrain + preflight")
print("=" * 70)
print(f"""
Champion (from DB): {get_current_champion()}
   The current champion is resolved from task6_champion_log at every
   predict + evaluate call. The pkl stores model weights only; its
   best_model_name field is a first-run fallback and NOT authoritative.
   Kernel restarts and file-cache regressions can no longer flip the
   serving champion.

DAILY USE (one command):

   run_task6_daily()          ← call once per morning

Automates:
  0. Preflight: refuse to predict if the production pkl holds prefit
     (leaky) calibrators — the 22 Sep incident cannot repeat silently
  1. Detect latest complete T1-4 date
  2. Predict with ALL 3 models (champion + 2 challengers, DB-resolved)
  3. Update actuals for matured predictions
  4. Print rolling 14-day F1 per model
  5. If Sunday: auto retrain, log swap to DB FIRST, then persist pkl weights

MANUAL FUNCTIONS (rarely needed):

   predict_task6(ticker, date, model)      — single (ticker, date, model)
   predict_all_tickers(date, store=T)      — full batch for one date
   update_actual_labels(prediction_date)   — fill actuals for one date
   evaluate_predictions(start, end)        — per-model F1 table
   retrain_weekly(retrain_lookback_days=28) — force retrain now
   t6_preflight()                          — check the production pkl
""")

# ── Startup check: is the production pkl safe right now? ────────────────────
print("Preflight on load:")
_ok_now = t6_preflight()
if not _ok_now:
    print("\n  ▶ Run  retrain_weekly()  in a new cell to restore the corrected models,")
    print("    then re-run this check with  t6_preflight().")
