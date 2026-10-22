# ═══════════════════════════════════════════════════════════════════════════
# R11 — T6 UPGRADE EXPERIMENT (pre-registered, read-only)
#
# Run AFTER the research cell (cell 32), which defines: cache, ALL_DATES, POS,
# TICKERS, FEATURE_NAMES_T6, _hdr, RESEARCH_DIR. Nothing here touches the
# production pickle, Supabase or the daily report.
#
# WHAT THIS ANSWERS
#   "Is there any version of Task 6 — different target, different features,
#    different model, or answering only when confident — that beats the naive
#    rules by enough to be worth putting in front of an investor?"
#
# TWO TARGETS, BOTH TESTED ON THE SAME ROWS
#   A  absolute   : 3-day return class (what production predicts today)
#   B  relative   : 3-day return MINUS the 15-stock average that day
#                   (the stock-selection question; the market move cancels)
#
# NINE CANDIDATES PER TARGET
#   global_majority   most common class in the training window        (baseline)
#   stock_majority    most common class for that stock                (baseline)
#   rev1_rule         yesterday's relative return, sign flipped       (baseline)
#   price_logreg / price_rf     4 price features only
#   news_logreg  / news_rf      the T6 feature set (T1-T4 evidence)
#   all_logreg   / all_rf       both
#
# ACCEPTANCE CRITERIA ARE FIXED BEFORE THE RESULTS ARE PRINTED (see below).
# A candidate is only worth promoting if it passes all of them. The verdict is
# computed by code, not chosen after looking at the table.
#
# WHAT THIS CANNOT TELL YOU
#   • Publication times. The cache does not prove an article was public by the
#     close of its date. Any positive result here still needs a timestamp audit
#     before it goes live.
#   • Out-of-sample truth. These are the same dates the earlier research used.
#     A pass here is a reason to run it forward on NEW dates, not a licence to
#     switch production.
# ═══════════════════════════════════════════════════════════════════════════
import os, json, warnings
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HORIZON        = 3        # same as production
CLASSES        = ["bearish", "bullish", "neutral"]
MIN_FIT_DATES  = 20       # training dates needed before the first test week
MIN_CAL_DATES  = 6        # calibration dates (never used to fit the model)
CAL_DATES      = 10
PX_HISTORY_PAD = 400      # extra calendar days of prices, so MOM60 is real
COST_PCT       = 0.30     # one-way cost assumption, % (report says 0.3-0.6%)

# ── Pre-registered acceptance criteria ─────────────────────────────────────
MIN_TEST_ROWS  = 300      # below this the test is inconclusive, not negative
F1_MARGIN      = 0.05     # macro F1 above the best baseline on the same rows
IC_T_MIN       = 2.0      # Newey-West t of the daily rank IC
SPREAD_MIN     = 0.30     # top5-bottom5 excess return per 3 days, %
SELECT_MARGIN  = 0.10     # accuracy edge on the top-confidence 20% of rows
HALF_SPLIT_BOTH = True    # the F1 edge must be positive in both halves

if "_hdr" not in globals():
    def _hdr(t):
        print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
RESEARCH_DIR = globals().get("RESEARCH_DIR",
                             "/app/outputs/research")
FEATURE_CACHE = globals().get(
    "FEATURE_CACHE", "/app/cache/t6_feature_rows.pkl")
T6_PKL = globals().get("ARTIFACTS_PATH_T6",
                       "/app/task6_artifacts_v1.pkl")
os.makedirs(RESEARCH_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 0. BOOTSTRAP — run with only the Setup cell if the research cell is not loaded
#
# The research cell (cell 32) leaves cache / ALL_DATES / TICKERS /
# FEATURE_NAMES_T6 in the session, but it also runs R1-R4, which takes far
# longer than this experiment. If those globals are absent (fresh kernel), this
# block rebuilds them: dates from Supabase, feature rows from the cache pickle
# the research cell already wrote to Drive, extracting only what is missing.
# ═══════════════════════════════════════════════════════════════════════════
_hdr("R11-0  BOOTSTRAP")
from sqlalchemy import text as _sqltext

if "_q" not in globals():
    def _q(sql, **p):
        with SUPABASE_ENGINE.connect() as conn:
            return conn.execute(_sqltext(sql), p).fetchall()

if "TICKERS" in globals():
    TICKERS = list(TICKERS)
elif "SGX_STOCKS" in globals():
    TICKERS = list(SGX_STOCKS.keys())
else:
    raise RuntimeError("Neither TICKERS nor SGX_STOCKS is defined — run the Setup cell.")
print(f"  Tickers: {len(TICKERS)}")

if "ALL_DATES" not in globals():
    if "SUPABASE_ENGINE" not in globals():
        raise RuntimeError("Neither ALL_DATES nor SUPABASE_ENGINE is in the session — "
                           "run the Setup cell first.")
    print("  ALL_DATES not in session — querying complete T1-T4 dates...")
    ALL_DATES = [str(r[0]) for r in _q("""
        SELECT DISTINCT n.date FROM news_articles n
        WHERE n.category = 'company'
          AND EXTRACT(ISODOW FROM CAST(n.date AS DATE)) <= 5
          AND EXISTS (SELECT 1 FROM technical_signals t
                      WHERE t.date = n.date AND t.ticker = n.ticker)
          AND EXISTS (SELECT 1 FROM macro_indicators m WHERE m.date = n.date)
          AND EXISTS (SELECT 1 FROM market_options o WHERE o.date = n.date)
          AND EXISTS (SELECT 1 FROM news_articles p
                      WHERE p.ticker='POLITICAL' AND p.date = n.date)
        ORDER BY n.date""")]
ALL_DATES = sorted(str(d) for d in ALL_DATES)
if not ALL_DATES:
    raise RuntimeError("No complete T1-T4 dates found.")
print(f"  Dates  : {ALL_DATES[0]} → {ALL_DATES[-1]}  ({len(ALL_DATES)})")

if "cache" not in globals():
    if os.path.exists(FEATURE_CACHE):
        import pickle as _pk
        with open(FEATURE_CACHE, "rb") as f:
            cache = _pk.load(f)
        print(f"  Loaded {len(cache)} cached feature rows from {FEATURE_CACHE}")
    else:
        cache = {}
        print(f"  No feature cache at {FEATURE_CACHE} — everything will be extracted.")

_todo = [(t, d) for d in ALL_DATES for t in TICKERS if (t, d) not in cache]
if _todo:
    if "extract_features_row" not in globals():
        print(f"  ⚠️  {len(_todo)} (ticker, date) rows are missing from the cache and "
              f"extract_features_row is not loaded.\n"
              f"     Running on the cached dates only. To include the rest, run T6 "
              f"Part A/B first, then this cell again.")
    else:
        import pickle as _pk
        os.makedirs(os.path.dirname(FEATURE_CACHE), exist_ok=True)
        print(f"  Extracting {len(_todo)} missing feature rows "
              f"(this is the only slow step)...")
        for _i, (_t, _d) in enumerate(_todo, 1):
            try:
                cache[(_t, _d)] = extract_features_row(_t, _d)
            except Exception:                                        # noqa: BLE001
                cache[(_t, _d)] = None
            if _i % 150 == 0:
                print(f"    {_i}/{len(_todo)}")
                with open(FEATURE_CACHE, "wb") as f:
                    _pk.dump(cache, f)
        with open(FEATURE_CACHE, "wb") as f:
            _pk.dump(cache, f)
        print(f"  Cache now holds {len(cache)} rows (saved to Drive).")

_have = sum(1 for k, v in cache.items() if v is not None)
print(f"  Usable feature rows: {_have} of {len(cache)}")
if _have < 100:
    raise RuntimeError("Fewer than 100 usable feature rows — run T6 Part A/B, then "
                       "re-run this cell so the missing rows can be extracted.")

if "FEATURE_NAMES_T6" not in globals():
    try:
        import pickle as _pk
        with open(T6_PKL, "rb") as f:
            FEATURE_NAMES_T6 = _pk.load(f)["feature_names"]
        print(f"  FEATURE_NAMES_T6 from {T6_PKL}: {len(FEATURE_NAMES_T6)} features")
    except Exception:                                                # noqa: BLE001
        _keys = Counter()
        for v in cache.values():
            if isinstance(v, dict):
                _keys.update(v.keys())
        FEATURE_NAMES_T6 = sorted(k for k, n in _keys.items()
                                  if n >= 0.9 * _have and not k.startswith("_"))
        print(f"  FEATURE_NAMES_T6 inferred from the cached rows: "
              f"{len(FEATURE_NAMES_T6)} features")
FEATURE_NAMES_T6 = list(FEATURE_NAMES_T6)
if not FEATURE_NAMES_T6:
    raise RuntimeError("Could not determine FEATURE_NAMES_T6.")

_hdr("R11  T6 UPGRADE EXPERIMENT — pre-registration")
print(f"""  Horizon           : {HORIZON} trading days
  Targets           : A absolute return class · B relative (vs 15-stock mean)
  Split             : weekly expanding walk-forward; a week's training rows
                      must have MATURED before that week's first date
  Calibration       : last {CAL_DATES} eligible dates, disjoint from the fit rows
  Label band        : +/- 0.5 x std of the target, frozen on the first
                      {MIN_FIT_DATES} dates and never re-estimated
  Promote a candidate only if ALL of these hold:
    1. macro F1 >= best baseline on the same rows + {F1_MARGIN:.2f}
    2. Brier score below that baseline's
    3. the F1 edge is positive in BOTH halves of the test period
    4. rank IC > 0 with Newey-West t > {IC_T_MIN:.1f}, and top5-bottom5 spread
       > {SPREAD_MIN:.2f}% per {HORIZON} days (above the {COST_PCT:.2f}% cost assumption)
    5. on the top 20% most confident rows, accuracy beats the majority class
       on those same rows by >= {SELECT_MARGIN:.2f}
  Fewer than {MIN_TEST_ROWS} test rows per candidate => INCONCLUSIVE, not a negative.""")


# ═══════════════════════════════════════════════════════════════════════════
# 1. PANEL — features + price features + matured forward returns
# ═══════════════════════════════════════════════════════════════════════════
_hdr("R11-1  PANEL")

# Long history on purpose. An earlier draft reused the research cell's CLOSES,
# which start at ALL_DATES[0] (~June), so every 60-day momentum value would
# have been missing and silently median-imputed. 400 extra days fixes that.
_start = (datetime.strptime(ALL_DATES[0], "%Y-%m-%d")
          - timedelta(days=PX_HISTORY_PAD)).strftime("%Y-%m-%d")
_end = (datetime.strptime(ALL_DATES[-1], "%Y-%m-%d")
        + timedelta(days=14)).strftime("%Y-%m-%d")
PXL = {}
for t in TICKERS:
    s = yf.Ticker(t).history(start=_start, end=_end)["Close"].sort_index()
    s.index = [str(x)[:10] for x in s.index]
    PXL[t] = s[~s.index.duplicated(keep="last")]
print(f"  Prices: {min(s.index[0] for s in PXL.values())} → "
      f"{max(s.index[-1] for s in PXL.values())}  "
      f"({sum(len(s) for s in PXL.values())} closes)")

PRICE_FEATURES = ["ret1", "ret5", "mom60_skip5", "vol20"]
NEWS_FEATURES = [c for c in FEATURE_NAMES_T6]
rows, _skip_hist, _skip_fwd, _skip_feat = [], 0, 0, 0
for t in TICKERS:
    s = PXL[t]
    idx, v = list(s.index), np.asarray(s.values, dtype=float)
    lr = np.diff(np.log(v), prepend=np.nan)
    pos = {d: j for j, d in enumerate(idx)}
    for d in ALL_DATES:
        j = pos.get(d)
        if j is None or j < 65:
            _skip_hist += 1
            continue
        if j + HORIZON >= len(idx):
            _skip_fwd += 1
            continue
        fd = cache.get((t, d))
        if fd is None:
            _skip_feat += 1
            continue
        r = {"ticker": t, "date": d, "maturity": idx[j + HORIZON],
             "fwd": (v[j + HORIZON] / v[j] - 1) * 100,
             "ret1": (v[j] / v[j - 1] - 1) * 100,
             "ret5": (v[j] / v[j - 5] - 1) * 100,
             "mom60_skip5": (v[j - 5] / v[j - 65] - 1) * 100,
             "vol20": float(np.nanstd(lr[j - 19:j + 1]) * 100)}
        r.update({c: fd.get(c, np.nan) for c in NEWS_FEATURES})
        rows.append(r)

panel = pd.DataFrame(rows)
if panel.empty:
    raise RuntimeError("Empty panel — no date has both a cached feature row and "
                       "a matured 3-day return.")
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
_num = PRICE_FEATURES + NEWS_FEATURES
panel[_num] = panel[_num].apply(pd.to_numeric, errors="coerce") \
                         .replace([np.inf, -np.inf], np.nan)

# cross-sectional versions: "how much more than the other stocks today"
panel["ex"] = panel["fwd"] - panel.groupby("date")["fwd"].transform("mean")
for c in _num:
    panel[c + "_x"] = panel[c] - panel.groupby("date")[c].transform("mean")
panel["week"] = pd.to_datetime(panel["date"]).dt.strftime("%G-W%V")

_dates = sorted(panel["date"].unique())
print(f"  Panel: {len(panel)} stock-days · {len(_dates)} dates "
      f"({_dates[0]} → {_dates[-1]}) · {panel.ticker.nunique()} stocks")
print(f"  Dropped: {_skip_hist} rows without 65 days of price history, "
      f"{_skip_fwd} not yet matured, {_skip_feat} without a cached feature row")
print(f"  MOM60 coverage {panel.mom60_skip5.notna().mean():.0%} · "
      f"news-feature coverage {panel[NEWS_FEATURES].notna().all(axis=1).mean():.0%}")
if panel.mom60_skip5.notna().mean() < 0.5:
    print("  ⚠️  MOM60 is mostly missing — the price-feature candidates are weakened.")

# frozen label bands (first MIN_FIT_DATES dates only)
_init = panel[panel["date"].isin(_dates[:MIN_FIT_DATES])]
THR = {"absolute": 0.5 * float(np.nanstd(_init["fwd"])),
       "relative": 0.5 * float(np.nanstd(_init["ex"]))}
print(f"  Frozen label bands: absolute +/-{THR['absolute']:.3f}%  ·  "
      f"relative +/-{THR['relative']:.3f}%   (from {_dates[0]} → {_dates[MIN_FIT_DATES-1]})")
for k, v in THR.items():
    if not np.isfinite(v) or v <= 0:
        raise RuntimeError(f"Bad {k} threshold {v} — check the price data.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _labels(x, thr):
    x = np.asarray(x, dtype=float)
    return np.where(x > thr, "bullish", np.where(x < -thr, "bearish", "neutral"))


def _nw_t(x, lags):
    """t-stat of the mean with Newey-West variance. Daily ICs from a 3-day
    horizon overlap, so a plain t-stat overstates significance."""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 5:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


def _proba_in_class_order(est, X):
    raw = est.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(est.classes_):
        out[:, CLASSES.index(str(c))] = raw[:, j]
    return out


def _fit_calibrated(base, fit, cal, cols, target_col, thr):
    """Fit on `fit`, calibrate on `cal`.

    `cal` rows are DISJOINT from `fit` rows and later in time, so cv="prefit"
    is correct here — unlike the production bug, where the calibrator was
    fitted on the model's own training rows.
    """
    yf_ = _labels(fit[target_col].to_numpy(), thr)
    yc_ = _labels(cal[target_col].to_numpy(), thr)
    if len(set(yf_)) < 3 or len(set(yc_)) < 2:
        return None
    model = clone(base).fit(fit[cols], yf_)
    try:
        from sklearn.frozen import FrozenEstimator            # sklearn >= 1.6
        est = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except ImportError:
        est = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    return est.fit(cal[cols], yc_)


def _model_specs(target):
    """Feature columns depend on the target: the relative target uses
    cross-sectionally demeaned features, which is the same reframe."""
    sfx = "_x" if target == "relative" else ""
    px = [c + sfx for c in PRICE_FEATURES]
    nw = [c + sfx for c in NEWS_FEATURES]
    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                       StandardScaler(),
                       LogisticRegression(C=0.1, max_iter=2000,
                                          class_weight="balanced", random_state=42))
    rf = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                       RandomForestClassifier(n_estimators=200, max_depth=4,
                                              min_samples_leaf=5, class_weight="balanced",
                                              random_state=42, n_jobs=-1))
    return {"price_logreg": (lr, px), "price_rf": (rf, px),
            "news_logreg":  (lr, nw), "news_rf":  (rf, nw),
            "all_logreg":   (lr, px + nw), "all_rf": (rf, px + nw)}


BASELINES = ["global_majority", "stock_majority", "rev1_rule"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════
def walk_forward(target):
    """One row per (model, ticker, date). Baselines and models see the exact
    same test rows, so every comparison below is on identical data."""
    tcol = "fwd" if target == "absolute" else "ex"
    thr = THR[target]
    rev_col = "ret1_x"                  # relative 1-day return (REV1)
    specs = _model_specs(target)
    out, weeks = [], sorted(panel["week"].unique())
    for wk in weeks:
        test = panel[panel["week"] == wk]
        if test.empty:
            continue
        wk_start = test["date"].min()
        # embargo: a training row's label must already be known on wk_start
        elig = panel[(panel["date"] < wk_start) & (panel["maturity"] < wk_start)]
        edates = sorted(elig["date"].unique())
        if len(edates) < MIN_FIT_DATES + MIN_CAL_DATES:
            continue
        n_cal = CAL_DATES if len(edates) >= MIN_FIT_DATES + CAL_DATES else MIN_CAL_DATES
        cal_start = edates[-n_cal]
        fit = elig[elig["maturity"] < cal_start]
        cal = elig[elig["date"] >= cal_start]
        if fit["date"].nunique() < MIN_FIT_DATES or cal["date"].nunique() < MIN_CAL_DATES:
            continue

        yfit = _labels(fit[tcol].to_numpy(), thr)
        gmaj = Counter(yfit).most_common(1)[0][0]
        smaj = {t: Counter(_labels(g[tcol].to_numpy(), thr)).most_common(1)[0][0]
                for t, g in fit.groupby("ticker")}
        # base rates give the baselines honest probabilities (a 0/1 vector
        # would make their Brier score look bad for no real reason)
        brate = {c: float((yfit == c).mean()) for c in CLASSES}
        srate = {t: {c: float((_labels(g[tcol].to_numpy(), thr) == c).mean())
                     for c in CLASSES} for t, g in fit.groupby("ticker")}
        rev_sd = float(np.nanstd(fit[rev_col])) or 1.0

        fitted = {}
        for name, (base, cols) in specs.items():
            try:
                fitted[name] = (_fit_calibrated(base, fit, cal, cols, tcol, thr), cols)
            except Exception as e:                                   # noqa: BLE001
                print(f"    ⚠️  {wk} {name}: {e!r}")

        # One prediction call per (week, model) instead of per row — a per-row
        # call is 10-20x slower with no difference in the numbers.
        base_cols = {"target": target, "week": wk, "cal_start": cal_start}
        meta = test[["date", "ticker", "maturity"]].reset_index(drop=True)
        meta["actual"] = _labels(test[tcol].to_numpy(), thr)
        meta["y"] = test[tcol].to_numpy(dtype=float)

        def _emit(name, preds, P, scores):
            d = meta.copy()
            d["model"], d["pred"], d["score"] = name, preds, scores
            for j, c in enumerate(CLASSES):
                d[f"p_{c}"] = P[:, j]
            for k, v in base_cols.items():
                d[k] = v
            out.append(d)

        n = len(meta)
        # baseline 1: one class for the whole market
        _emit("global_majority", np.repeat(gmaj, n),
              np.tile([brate[c] for c in CLASSES], (n, 1)), np.full(n, np.nan))
        # baseline 2: each stock's own most common class
        _emit("stock_majority",
              np.array([smaj.get(t, gmaj) for t in meta["ticker"]]),
              np.array([[srate.get(t, brate)[c] for c in CLASSES] for t in meta["ticker"]]),
              np.full(n, np.nan))
        # baseline 3: REV1 — yesterday's relative return, sign flipped
        z = -test[rev_col].to_numpy(dtype=float) / rev_sd
        pr = np.where(z > 0.5, "bullish", np.where(z < -0.5, "bearish", "neutral"))
        conf = np.clip(np.abs(z) / 2.0, 1.0 / 3, 0.98)
        Pr = np.zeros((n, len(CLASSES)))
        for j, c in enumerate(CLASSES):
            Pr[:, j] = np.where(pr == c, conf, (1 - conf) / 2)
        _emit("rev1_rule", pr, Pr, z)
        # candidate models
        for name, (est, cols) in fitted.items():
            if est is None:
                continue
            P = _proba_in_class_order(est, test[cols])
            _emit(name, np.array(CLASSES)[P.argmax(axis=1)], P,
                  P[:, CLASSES.index("bullish")] - P[:, CLASSES.index("bearish")])
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    return res[res["score"].notna() | ~res["model"].eq("rev1_rule")]


# ═══════════════════════════════════════════════════════════════════════════
# 4. SCORING
# ═══════════════════════════════════════════════════════════════════════════
def score(pred):
    """Every model is scored on the intersection of rows all models produced,
    so a week one model skipped cannot flatter or punish another."""
    keys = None
    for m, g in pred.groupby("model"):
        k = set(zip(g["date"], g["ticker"]))
        keys = k if keys is None else (keys & k)
    pred = pred[[(d, t) in keys for d, t in zip(pred["date"], pred["ticker"])]].copy()
    dates = sorted(pred["date"].unique())
    half = dates[len(dates) // 2] if len(dates) > 3 else None
    recs = []
    for m, g in pred.groupby("model"):
        P = g[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=float)
        T = np.zeros_like(P)
        for j, c in enumerate(CLASSES):
            T[:, j] = (g["actual"].to_numpy() == c).astype(float)
        maj_rows = g["actual"].value_counts(normalize=True).max()
        # rank IC per day + Newey-West t, and the top5/bottom5 spread
        ics, spreads = [], []
        if g["score"].notna().any():
            for d, gd in g.groupby("date"):
                gd = gd.dropna(subset=["score", "y"])
                if len(gd) < 8 or gd["score"].nunique() < 2:
                    continue
                ic = gd["score"].rank().corr(gd["y"].rank())
                if np.isfinite(ic):
                    ics.append(ic)
                k = min(5, len(gd) // 3)
                if k >= 3:
                    o = gd.sort_values("score")
                    spreads.append(o["y"].tail(k).mean() - o["y"].head(k).mean())
        # selective prediction: the most confident 20% of rows
        conf = P.max(axis=1)
        cut = np.quantile(conf, 0.80) if len(conf) > 10 else np.inf
        sel = g[conf >= cut]
        sel_acc = (sel["pred"] == sel["actual"]).mean() if len(sel) else np.nan
        sel_maj = sel["actual"].value_counts(normalize=True).max() if len(sel) else np.nan
        rec = {"model": m, "n": len(g), "days": g["date"].nunique(),
               "accuracy": accuracy_score(g["actual"], g["pred"]),
               "macro_f1": f1_score(g["actual"], g["pred"], average="macro",
                                    labels=CLASSES, zero_division=0),
               "bal_acc": balanced_accuracy_score(g["actual"], g["pred"]),
               "brier": float(np.mean(np.sum((P - T) ** 2, axis=1))),
               "always_majority": maj_rows,
               "IC": np.mean(ics) if ics else np.nan,
               "IC_t": _nw_t(ics, HORIZON) if len(ics) > 4 else np.nan,
               "IC_pos%": float(np.mean(np.array(ics) > 0)) if ics else np.nan,
               "spread%": np.mean(spreads) if spreads else np.nan,
               "sel20_acc": sel_acc, "sel20_majority": sel_maj,
               "bull%": float((g["pred"] == "bullish").mean()),
               "bear%": float((g["pred"] == "bearish").mean()),
               "neu%": float((g["pred"] == "neutral").mean())}
        for lbl, sub in (("h1", g[g["date"] < half] if half else g.iloc[:0]),
                         ("h2", g[g["date"] >= half] if half else g.iloc[:0])):
            rec[f"f1_{lbl}"] = (f1_score(sub["actual"], sub["pred"], average="macro",
                                         labels=CLASSES, zero_division=0)
                                if len(sub) else np.nan)
        recs.append(rec)
    return pd.DataFrame(recs).set_index("model"), pred


def verdict(tab, target):
    """Apply the pre-registered criteria. Code decides, not the reader."""
    base = tab.loc[[b for b in BASELINES if b in tab.index]]
    best_b = base["macro_f1"].idxmax()
    bf1 = base.loc[best_b, "macro_f1"]
    bbrier = base["brier"].min()                 # hardest baseline to beat, per metric
    bh1, bh2 = base.loc[best_b, "f1_h1"], base.loc[best_b, "f1_h2"]
    print(f"\n  Bar to clear: macro F1 {bf1:.3f} ({best_b}) · "
          f"Brier {bbrier:.3f} ({base['brier'].idxmin()}) · "
          f"accuracy {base['accuracy'].max():.3f} ({base['accuracy'].idxmax()})")
    if best_b == "rev1_rule":
        print("  Note: the short-term reversal rule is the strongest baseline here — the"
              "\n  same effect R6 found over 2019-2025. Worth saying so in the report.")
    out = []
    for m in tab.index:
        if m in BASELINES:
            continue
        r = tab.loc[m]
        c = {
            "1 F1 edge":   r["macro_f1"] >= bf1 + F1_MARGIN,
            "2 Brier":     r["brier"] < bbrier,
            "3 both halves": (not HALF_SPLIT_BOTH) or
                             (np.isfinite(r["f1_h1"]) and np.isfinite(r["f1_h2"])
                              and r["f1_h1"] > bh1 and r["f1_h2"] > bh2),
            "4 IC & spread": (np.isfinite(r["IC_t"]) and r["IC"] > 0
                              and r["IC_t"] > IC_T_MIN
                              and np.isfinite(r["spread%"]) and r["spread%"] > SPREAD_MIN),
            "5 selective": (np.isfinite(r["sel20_acc"]) and np.isfinite(r["sel20_majority"])
                            and r["sel20_acc"] - r["sel20_majority"] >= SELECT_MARGIN),
        }
        enough = r["n"] >= MIN_TEST_ROWS
        res = ("PROMOTE-CANDIDATE" if all(c.values()) and enough else
               "INCONCLUSIVE (too few rows)" if not enough else "REJECT")
        out.append({"target": target, "model": m, "verdict": res,
                    **{k: bool(v) for k, v in c.items()},
                    "macro_f1": r["macro_f1"], "baseline_f1": bf1,
                    "edge": r["macro_f1"] - bf1, "n": int(r["n"])})
    V = pd.DataFrame(out).sort_values("edge", ascending=False)
    print(V.to_string(index=False))
    return V


# ═══════════════════════════════════════════════════════════════════════════
# 5. RUN
# ═══════════════════════════════════════════════════════════════════════════
pd.set_option("display.width", 220, "display.max_columns", 40,
              "display.float_format", lambda x: f"{x:.3f}")
ALL_PRED, ALL_TAB, ALL_V = [], {}, []
for target in ("absolute", "relative"):
    _hdr(f"R11-{'2' if target == 'absolute' else '3'}  TARGET = {target.upper()}"
         + ("   (what production predicts today)" if target == "absolute"
            else "   (stock selection: beat the other 14)"))
    pr = walk_forward(target)
    if pr.empty:
        print("  No test rows produced — not enough matured history yet.")
        continue
    tab, pr = score(pr)
    print(f"  Test rows: {pr['date'].nunique()} dates "
          f"({pr['date'].min()} → {pr['date'].max()}), "
          f"{pr['week'].nunique()} weeks, {pr['ticker'].nunique()} stocks, "
          f"label band +/-{THR[target]:.3f}%")
    keep = ["n", "days", "accuracy", "macro_f1", "bal_acc", "brier", "always_majority",
            "IC", "IC_t", "spread%", "sel20_acc", "sel20_majority",
            "bull%", "bear%", "neu%", "f1_h1", "f1_h2"]
    order = [m for m in BASELINES if m in tab.index] + \
            [m for m in tab.index if m not in BASELINES]
    print(tab.loc[order, keep].to_string())
    ALL_V.append(verdict(tab, target))
    ALL_PRED.append(pr)
    ALL_TAB[target] = tab

# ═══════════════════════════════════════════════════════════════════════════
# 6. DECISION
# ═══════════════════════════════════════════════════════════════════════════
_hdr("R11-4  DECISION")
if ALL_V:
    V = pd.concat(ALL_V, ignore_index=True)
    win = V[V["verdict"] == "PROMOTE-CANDIDATE"]
    if len(win):
        print("  Candidates that passed every pre-registered criterion:")
        print(win[["target", "model", "macro_f1", "baseline_f1", "edge", "n"]].to_string(index=False))
        print("""
  NEXT STEP — and it is not "switch production". Freeze this configuration in
  writing (target, features, model, thresholds), then run it forward on dates
  that did not exist when this cell ran. Only a forward record of 4+ weeks that
  keeps the edge justifies replacing the deployed T6.""")
    else:
        print("""  No candidate passed. This is a result, not a failure: with this data
  (15 stocks, one quarter of complete history, a 3-day horizon) neither the T1-T4
  evidence features nor price features beat the naive rules by a usable margin,
  on either the absolute or the relative target.

  Report it that way. The honest version is more defensible than a tuned number:
    • say which targets and feature sets were tested, and on how many rows
    • say the criteria were fixed before the results were seen (they are printed
      at the top of this cell's output)
    • keep production T6 as it is, keep the suppression control, and keep the
      reliability wording that already tells the reader not to trade on it""")
    for tgt in ALL_TAB:
        t = ALL_TAB[tgt]
        b = t.loc[[x for x in BASELINES if x in t.index], "macro_f1"].max()
        m = t.loc[[x for x in t.index if x not in BASELINES], "macro_f1"].max()
        print(f"\n  {tgt:<9}: best baseline F1 {b:.3f} · best model F1 {m:.3f} · "
              f"edge {m - b:+.3f}  (needed {F1_MARGIN:+.2f})")
    try:
        P = pd.concat(ALL_PRED, ignore_index=True)
        P.to_csv(f"{RESEARCH_DIR}/R11_predictions.csv", index=False)
        pd.concat([ALL_TAB[k].assign(target=k).reset_index() for k in ALL_TAB],
                  ignore_index=True).to_csv(f"{RESEARCH_DIR}/R11_summary.csv", index=False)
        V.to_csv(f"{RESEARCH_DIR}/R11_verdicts.csv", index=False)
        print(f"\n  Saved R11_predictions.csv / R11_summary.csv / R11_verdicts.csv in {RESEARCH_DIR}")
    except Exception as e:                                           # noqa: BLE001
        print(f"  (could not save CSVs: {e!r})")
else:
    print("  Nothing to decide — no target produced test rows.")
print("\n  Read-only: production models, Supabase and the daily report are untouched.")
print("  Save this output to the research log.")