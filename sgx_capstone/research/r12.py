# ═══════════════════════════════════════════════════════════════════════════
# R12 — PRE-REGISTERED FORWARD TEST OF THE ONE SURVIVING HYPOTHESIS
#
# WHY THIS EXISTS
#   R11 tested 6 models x 2 targets = 12 combinations. Every one failed the
#   pre-registered bar. But one number kept showing up: the T1-T4 feature
#   score RANKS the 15 stocks slightly better than chance —
#       a weak positive cross-sectional rank IC was observed on 22 dates.
#   Its exact values must be read from R11_summary.csv; selecting the best
#   of 12 combinations can easily produce an apparent positive result by chance. It is a HYPOTHESIS, not a finding.
#
#   The only way to turn it into evidence is to freeze it and run it forward on
#   dates that did not exist when the hypothesis was formed. That is what this
#   cell sets up. Nothing here changes production T6, the daily report, or any
#   Supabase table — it writes its own log file.
#
# WHAT IS FROZEN (and written to R12_config.json, so it can be shown to be
# unchanged later):
#   target      3-day return relative to the 15-stock average that day
#   features    the T6 feature set, cross-sectionally demeaned each day
#   model       LogisticRegression(C=0.1, class_weight='balanced'), no calibration
#   score       p(bullish) - p(bearish)   ← a RANKING score, not a class label
#   decision    rank the 15: top 3 = "prefer", bottom 3 = "avoid", rest "no view"
#   refit       weekly, on every row whose label had matured by then
#
# ACCEPTANCE (fixed now, checked only after the data arrives):
#   >= 15 new trading dates, and
#   rank IC > 0 with Newey-West t > 2.0, and
#   prefer-minus-avoid spread > 0.30% per 3 days (the cost assumption)
#   → then, and only then, T6 may publish a ranking. Otherwise the report says
#     the system has no demonstrated predictive value, which is where it stands.
#
# WHY A RANKING AND NOT A CLASS LABEL
#   R11's honest finding: after out-of-fold calibration, every model predicts
#   "neutral" for 100% of rows. That is correct behaviour — neutral holds 55%
#   of the labels and the signal is weak, so a calibrated model should never
#   pick a side. A class label therefore cannot carry what little information
#   exists; a cross-sectional ranking can.
#
# DAILY USE, after running this cell once:
#   t6_rank_today()        — score today's 15 stocks, append to the forward log
#   t6_rank_scoreboard()   — the out-of-sample record so far
#   t6_rank_refit()        — weekly (Sunday), refit on newly matured rows
# ═══════════════════════════════════════════════════════════════════════════
import os, json, pickle, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import text as _sqltext
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJ          = "/app"
R12_DIR       = f"{PROJ}/outputs/research"
R12_MODEL     = f"{PROJ}/task6_rank_model_v1.pkl"
R12_CONFIG    = f"{R12_DIR}/R12_config.json"
R12_LOG       = f"{R12_DIR}/R12_forward_log.csv"
R12_REFITS    = f"{R12_DIR}/R12_refit_log.csv"
FEATURE_CACHE = globals().get("FEATURE_CACHE", f"{PROJ}/cache/t6_feature_rows.pkl")
T6_PKL        = globals().get("ARTIFACTS_PATH_T6", f"{PROJ}/task6_artifacts_v1.pkl")

HORIZON       = 3
N_SIDE        = 3         # top 3 "prefer", bottom 3 "avoid", of 15
WARMUP_DATES  = 20        # dates used to freeze the label band
COST_PCT      = 0.30
ACCEPT_DATES  = 15
ACCEPT_IC_T   = 2.0
ACCEPT_SPREAD = 0.30
PX_PAD        = 400
os.makedirs(R12_DIR, exist_ok=True)


def _hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def _q(sql, **p):
    with SUPABASE_ENGINE.connect() as conn:
        return conn.execute(_sqltext(sql), p).fetchall()


def _nw_t(x, lags=HORIZON):
    """t-stat of the mean, Newey-West. Overlapping 3-day returns make daily
    ICs autocorrelated; a plain t-stat would overstate significance."""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 5:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


def _tickers():
    if "SGX_STOCKS" in globals():
        return list(SGX_STOCKS.keys())
    if "TICKERS" in globals():
        return list(TICKERS)
    raise RuntimeError("SGX_STOCKS is not defined — run the Setup cell first.")


def _prices(tickers, start, end):
    out = {}
    for t in tickers:
        s = yf.Ticker(t).history(start=start, end=end)["Close"].sort_index()
        s.index = [str(x)[:10] for x in s.index]
        out[t] = s[~s.index.duplicated(keep="last")]
    return out


def _complete_dates():
    if "ALL_DATES" in globals() and ALL_DATES:
        return sorted(str(d) for d in ALL_DATES)
    return [str(r[0]) for r in _q("""
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


def _feature_names(cache):
    if "FEATURE_NAMES_T6" in globals():
        return list(FEATURE_NAMES_T6)
    try:
        with open(T6_PKL, "rb") as f:
            return list(pickle.load(f)["feature_names"])
    except Exception:                                                # noqa: BLE001
        from collections import Counter
        k = Counter()
        for v in cache.values():
            if isinstance(v, dict):
                k.update(v.keys())
        n = max(1, sum(1 for v in cache.values() if isinstance(v, dict)))
        return sorted(c for c, m in k.items() if m >= 0.9 * n and not c.startswith("_"))


def _training_panel():
    """Rebuild the training panel from the feature cache and prices.

    Independent of R11 on purpose: the weekly refit has to work on any kernel,
    not only one where a research cell happens to be loaded.
    """
    tickers = _tickers()
    dates = _complete_dates()
    if not dates:
        raise RuntimeError("No complete T1-T4 dates found.")
    if "cache" in globals() and globals()["cache"]:
        cache = globals()["cache"]
    elif os.path.exists(FEATURE_CACHE):
        with open(FEATURE_CACHE, "rb") as f:
            cache = pickle.load(f)
    else:
        raise RuntimeError(f"No feature cache at {FEATURE_CACHE} — run the research "
                           f"cell or T6 Part A/B once so the rows get cached.")
    feats = _feature_names(cache)
    start = (datetime.strptime(dates[0], "%Y-%m-%d")
             - timedelta(days=PX_PAD)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    px = _prices(tickers, start, end)

    rows = []
    for t in tickers:
        s = px[t]
        idx, v = list(s.index), np.asarray(s.values, dtype=float)
        pos = {d: j for j, d in enumerate(idx)}
        for d in dates:
            j = pos.get(d)
            fd = cache.get((t, d))
            if j is None or fd is None or j + HORIZON >= len(idx):
                continue
            r = {"ticker": t, "date": d, "maturity": idx[j + HORIZON],
                 "fwd": (v[j + HORIZON] / v[j] - 1) * 100}
            r.update({c: fd.get(c, np.nan) for c in feats})
            rows.append(r)
    P = pd.DataFrame(rows)
    if P.empty:
        raise RuntimeError("Empty training panel.")
    P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    P[feats] = P[feats].apply(pd.to_numeric, errors="coerce") \
                       .replace([np.inf, -np.inf], np.nan)
    P["ex"] = P["fwd"] - P.groupby("date")["fwd"].transform("mean")
    for c in feats:
        P[c + "_x"] = P[c] - P.groupby("date")[c].transform("mean")
    return P, feats


def _new_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                           random_state=42))


def _labels(x, thr):
    x = np.asarray(x, dtype=float)
    return np.where(x > thr, "bullish", np.where(x < -thr, "bearish", "neutral"))


# ═══════════════════════════════════════════════════════════════════════════
# SETUP — freeze the configuration and fit the first model
# ═══════════════════════════════════════════════════════════════════════════
def r12_setup(force=False):
    _hdr("R12 SETUP — freeze the hypothesis")
    if os.path.exists(R12_CONFIG) and not force:
        with open(R12_CONFIG) as f:
            cfg = json.load(f)
        print(f"  Already frozen on {cfg['frozen_at']} — not touching it.")
        print(f"  Config: {R12_CONFIG}")
        print("  (force=True would overwrite it, which ends the forward test's "
              "validity — don't, unless nothing has been logged yet.)")
        return cfg

    P, feats = _training_panel()
    dates = sorted(P["date"].unique())
    warm = P[P["date"].isin(dates[:WARMUP_DATES])]
    band = 0.5 * float(np.nanstd(warm["ex"]))
    xcols = [c + "_x" for c in feats]
    y = _labels(P["ex"].to_numpy(), band)
    model = _new_model().fit(P[xcols], y)

    # Read the actual prior experiment, rather than hard-coding numbers from
    # a different run into the frozen provenance.
    prior_path = f"{R12_DIR}/R11_summary.csv"
    prior = pd.read_csv(prior_path)
    picked = prior[(prior["target"] == "relative") &
                   (prior["model"] == "news_logreg")]
    if len(picked) != 1:
        raise RuntimeError("R11 relative/news_logreg row missing or ambiguous")
    picked = picked.iloc[0]
    prior_statement = (f"Selected after comparing R11 candidates: "
                       f"IC {picked['IC']:.3f}, NW t {picked['IC_t']:.2f}, "
                       f"spread {picked['spread%']:.3f}%/3d on "
                       f"{int(picked['days'])} dates. This is hypothesis "
                       "selection, not independent validation.")
    cfg = {
        "frozen_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hypothesis": ("The T1-T4 feature score ranks the 15 SGX stocks better "
                       "than chance on 3-day relative return."),
        "provenance": prior_statement,
        "target": f"{HORIZON}-day return minus the {len(_tickers())}-stock daily mean",
        "label_band_pct": band,
        "label_band_from": f"{dates[0]}..{dates[min(WARMUP_DATES, len(dates)) - 1]}",
        "features": feats, "feature_cols": xcols,
        "model": "LogisticRegression(C=0.1, class_weight=balanced), no calibration",
        "score": "p(bullish) - p(bearish)",
        "decision_rule": f"top {N_SIDE} = prefer, bottom {N_SIDE} = avoid, rest = no view",
        "refit": "weekly, on rows matured by the refit date",
        "acceptance": {"min_dates": ACCEPT_DATES, "min_IC_t": ACCEPT_IC_T,
                       "min_spread_pct": ACCEPT_SPREAD, "cost_pct": COST_PCT},
        "trained_through": max(P["maturity"]), "n_train_rows": int(len(P)),
        "train_dates": f"{dates[0]}..{dates[-1]}",
        "model_file": R12_MODEL, "log_file": R12_LOG,
    }
    with open(R12_MODEL, "wb") as f:
        pickle.dump({"model": model, "config": cfg}, f)
        f.flush(); os.fsync(f.fileno())
    with open(R12_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Training rows   : {len(P)}  ({dates[0]} → {dates[-1]}, {len(dates)} dates)")
    print(f"  Label band      : +/-{band:.3f}%  (frozen on {cfg['label_band_from']})")
    print(f"  Features        : {len(feats)}, cross-sectionally demeaned")
    print(f"  Class balance   : " + "  ".join(f"{c} {(y == c).mean():.0%}"
                                              for c in ("bullish", "neutral", "bearish")))
    print(f"  Model saved     : {R12_MODEL}")
    print(f"  Config saved    : {R12_CONFIG}")
    print(f"""
  From today, the forward record is the only evidence that counts. The
  in-sample numbers in R11 are how this hypothesis was chosen, so they cannot
  also be the test of it — say that in the report, in those words.""")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# DAILY — score today's 15 stocks and append to the forward log
# ═══════════════════════════════════════════════════════════════════════════
def t6_rank_today(date=None, store=True, verbose=True):
    """Rank the 15 stocks for one date and append to the forward log.

    Uses only that day's own rows for the cross-sectional demeaning, so
    nothing from the future enters the score. Idempotent per date.
    """
    with open(R12_MODEL, "rb") as f:
        art = pickle.load(f)
    model, cfg = art["model"], art["config"]
    feats, xcols = cfg["features"], cfg["feature_cols"]
    tickers = _tickers()

    if date is None:
        if "get_latest_complete_data_date" in globals():
            date = get_latest_complete_data_date()
        if not date:
            date = sorted(_complete_dates())[-1]
    date = str(date)[:10]

    if "extract_features_row" not in globals():
        raise RuntimeError("extract_features_row is not defined — run T6 Part A/B first.")
    rows = {}
    for t in tickers:
        try:
            rows[t] = extract_features_row(t, date)
        except Exception as e:                                        # noqa: BLE001
            if verbose:
                print(f"  ⚠️  {t}: {e!r}")
    rows = {t: r for t, r in rows.items() if r}
    if len(rows) < 8:
        print(f"  🛑 Only {len(rows)} of {len(tickers)} stocks have features for {date}. "
              f"Not scoring — a cross-sectional ranking needs the whole group.")
        return None

    X = pd.DataFrame([{c: r.get(c, np.nan) for c in feats} for r in rows.values()],
                     index=list(rows))
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xx = pd.DataFrame({c + "_x": X[c] - X[c].mean() for c in feats}, index=X.index)
    p = model.predict_proba(Xx[xcols])
    cls = list(model.classes_)
    score = p[:, cls.index("bullish")] - p[:, cls.index("bearish")]

    out = pd.DataFrame({"date": date, "ticker": X.index, "score": score})
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    n = len(out)
    out["decision"] = ["prefer" if i < N_SIDE else "avoid" if i >= n - N_SIDE
                       else "no view" for i in range(n)]
    out["model_frozen_at"] = cfg["frozen_at"]
    out["logged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if store:
        old = pd.read_csv(R12_LOG) if os.path.exists(R12_LOG) else pd.DataFrame()
        if len(old) and (old["date"].astype(str) == date).any():
            print(f"  ⏭️  {date} already in the forward log — not appending again.")
        else:
            pd.concat([old, out], ignore_index=True).to_csv(R12_LOG, index=False)
            if verbose:
                print(f"  ✅ appended {len(out)} rows for {date} → {R12_LOG}")
    if verbose:
        print(f"\n  Ranking for {date}  (score = p_bullish - p_bearish, "
              f"cross-sectional)")
        print(out[["rank", "ticker", "score", "decision"]].to_string(index=False))
        print("\n  This is a RELATIVE ranking under test, not a forecast. It says"
              "\n  nothing about whether the market goes up, and it has no verified"
              "\n  track record yet — see t6_rank_scoreboard().")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SCOREBOARD — the out-of-sample record
# ═══════════════════════════════════════════════════════════════════════════
def t6_rank_scoreboard(verbose=True):
    if not os.path.exists(R12_LOG):
        print("  Forward log is empty — run t6_rank_today() first.")
        return None
    L = pd.read_csv(R12_LOG)
    L["date"] = L["date"].astype(str).str[:10]
    tickers = sorted(L["ticker"].unique())
    start = (datetime.strptime(L["date"].min(), "%Y-%m-%d")
             - timedelta(days=10)).strftime("%Y-%m-%d")
    px = _prices(tickers, start, (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"))

    recs = []
    for t, g in L.groupby("ticker"):
        s = px.get(t)
        if s is None or not len(s):
            continue
        idx, v = list(s.index), np.asarray(s.values, dtype=float)
        pos = {d: j for j, d in enumerate(idx)}
        for r in g.itertuples(index=False):
            j = pos.get(r.date)
            if j is None or j + HORIZON >= len(idx):
                continue
            recs.append({"date": r.date, "ticker": t, "score": float(r.score),
                         "decision": r.decision, "rank": int(r.rank),
                         "fwd": (v[j + HORIZON] / v[j] - 1) * 100,
                         "maturity": idx[j + HORIZON]})
    M = pd.DataFrame(recs)
    _hdr("R12 FORWARD SCOREBOARD")
    print(f"  Logged      : {L['date'].nunique()} dates "
          f"({L['date'].min()} → {L['date'].max()})")
    if M.empty:
        print(f"  Matured     : 0 — the first {HORIZON}-day return is not in yet. "
              f"Check back in {HORIZON} trading days.")
        return None
    M["ex"] = M["fwd"] - M.groupby("date")["fwd"].transform("mean")

    ics, spreads, hits = [], [], []
    for d, g in M.groupby("date"):
        if g["ticker"].nunique() < 8 or g["score"].nunique() < 2:
            continue
        ic = g["score"].rank().corr(g["ex"].rank())
        if np.isfinite(ic):
            ics.append(ic)
        pref, avoid = g[g.decision == "prefer"], g[g.decision == "avoid"]
        if len(pref) and len(avoid):
            sp = pref["ex"].mean() - avoid["ex"].mean()
            spreads.append(sp)
            hits.append(sp > 0)
    nd = M["date"].nunique()
    ic_m = float(np.mean(ics)) if ics else np.nan
    ic_t = _nw_t(ics) if len(ics) > 4 else np.nan
    sp_m = float(np.mean(spreads)) if spreads else np.nan

    print(f"  Matured     : {nd} dates, {len(M)} stock-days "
          f"({M['date'].min()} → {M['date'].max()})")
    print(f"  rank IC     : {ic_m:+.3f}   Newey-West t {ic_t:+.2f}   "
          f"positive on {np.mean(np.array(ics) > 0):.0%} of days" if ics else
          "  rank IC     : n/a")
    print(f"  prefer-avoid: {sp_m:+.3f}% per {HORIZON} days   "
          f"right way round on {np.mean(hits):.0%} of days" if spreads else
          "  prefer-avoid: n/a")
    print(f"  after {COST_PCT:.2f}% cost: {sp_m - COST_PCT:+.3f}%" if spreads else "")

    ok = {f"1 at least {ACCEPT_DATES} dates": nd >= ACCEPT_DATES,
          "2 IC > 0": bool(np.isfinite(ic_m) and ic_m > 0),
          f"3 NW t > {ACCEPT_IC_T}": bool(np.isfinite(ic_t) and ic_t > ACCEPT_IC_T),
          f"4 spread > {ACCEPT_SPREAD}%": bool(np.isfinite(sp_m) and sp_m > ACCEPT_SPREAD)}
    print("\n  Pre-registered acceptance (frozen before any of this data existed):")
    for k, v in ok.items():
        print(f"    {'✅' if v else '❌'}  {k}")
    if all(ok.values()):
        print("""
  → PASSED. T6 may publish the ranking, worded as a relative preference with
    this record attached (dates, IC, spread, cost). It is still not a
    probability of any stock rising.""")
    elif nd < ACCEPT_DATES:
        print(f"\n  → NOT YET. {ACCEPT_DATES - nd} more trading dates needed. "
              f"Keep running t6_rank_today() daily; do not change the config.")
    else:
        print("""
  → FAILED. Report it: the hypothesis was frozen, run forward, and did not
    hold. T6 keeps its current wording — no demonstrated predictive value.
    A negative result from a pre-registered test is a real finding; a positive
    result from a test chosen after the fact is not.""")
    try:
        M.to_csv(f"{R12_DIR}/R12_forward_matured.csv", index=False)
        pd.DataFrame({"date": sorted(M["date"].unique())[:len(ics)],
                      "ic": ics}).to_csv(f"{R12_DIR}/R12_forward_ic.csv", index=False)
    except Exception:                                                # noqa: BLE001
        pass
    return M


# ═══════════════════════════════════════════════════════════════════════════
# WEEKLY REFIT — same configuration, more data
# ═══════════════════════════════════════════════════════════════════════════
def t6_rank_refit():
    """Refit the frozen model on every row matured by today. The configuration
    (target, features, band, decision rule) does NOT change — only the data."""
    _hdr("R12 WEEKLY REFIT")
    with open(R12_MODEL, "rb") as f:
        cfg = pickle.load(f)["config"]
    P, feats = _training_panel()
    if list(feats) != list(cfg["features"]):
        print("  🛑 The feature set changed since the config was frozen. Refusing to"
              "\n     refit: that would silently redefine the experiment. Either keep"
              "\n     the old feature set, or start a new config and a new log.")
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    P = P[P["maturity"] < today]
    y = _labels(P["ex"].to_numpy(), cfg["label_band_pct"])
    model = _new_model().fit(P[cfg["feature_cols"]], y)
    cfg = {**cfg, "refit_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "n_train_rows": int(len(P)), "trained_through": max(P["maturity"])}
    with open(R12_MODEL, "wb") as f:
        pickle.dump({"model": model, "config": cfg}, f)
        f.flush(); os.fsync(f.fileno())
    row = pd.DataFrame([{"refit_at": cfg["refit_at"], "n_train_rows": len(P),
                         "trained_through": cfg["trained_through"],
                         "label_band_pct": cfg["label_band_pct"]}])
    old = pd.read_csv(R12_REFITS) if os.path.exists(R12_REFITS) else pd.DataFrame()
    pd.concat([old, row], ignore_index=True).to_csv(R12_REFITS, index=False)
    print(f"  Refitted on {len(P)} rows through {cfg['trained_through']} "
          f"(band unchanged at +/-{cfg['label_band_pct']:.3f}%)")
    print(f"  Logged to {R12_REFITS}")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# RUN: freeze it now, score today, show the (empty) scoreboard
# ═══════════════════════════════════════════════════════════════════════════
_cfg = r12_setup()
_hdr("R12 TODAY")
try:
    t6_rank_today()
except Exception as _e:                                              # noqa: BLE001
    print(f"  Could not score today: {_e!r}")
    print("  Run T6 Part A/B (for extract_features_row), then t6_rank_today().")
t6_rank_scoreboard()

print(f"""
{'=' * 78}
DAILY ROUTINE FROM TOMORROW
{'=' * 78}
  run_task6_daily()        production T6, unchanged
  t6_rank_today()          append today's ranking to the forward log
  t6_rank_scoreboard()     the record so far (and whether it passed)
  t6_rank_refit()          Sundays only

  Do not edit R12_config.json, the label band, the feature list or the
  decision rule while the test runs. If something must change, start a new
  config with a new name and say in the report that the first test was
  abandoned and why. That sentence is worth more than a tuned number.""")