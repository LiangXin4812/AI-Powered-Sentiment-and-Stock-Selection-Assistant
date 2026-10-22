# ═══════════════════════════════════════════════════════════════════
# TASK 6 / TASK 5 — RESEARCH CELL  (research directions + fixes)
#
# Paste at the END of the T6 section, after Part B and the retrain_weekly v2
# cell. Needs in session: Setup (SUPABASE_ENGINE, SGX_STOCKS), T6 Part A/B
# (extract_features_row, FEATURE_NAMES_T6). Read-only: it never writes to
# Supabase or to the production pkl. Results are saved as CSV in RESEARCH_DIR.
#
#   R1  Per-stock model consistency   — do LogReg / RandomF / XGBoost agree
#                                       on each stock (target >= 80%), and is
#                                       the call better when they agree?
#   R2  Horizon robustness            — walk-forward retrain with
#                                       FORWARD_HORIZON in {1, 3, 5}, each with
#                                       the OLD (leaky) and NEW (out-of-fold)
#                                       calibration, vs each stock's naive rule
#   R3  Retrain window sensitivity    — champion selection replayed with
#                                       lookback {7,14,21,28,60} x margin
#                                       {0, 0.03, 0.05}: accuracy and swaps
#   R4  T5 evidence-layer replay      — rl_v7 net scores re-scored under
#                                       variants that remove the bullish tilt
#
# Runtime: first run ~10-20 min (feature extraction, cached to Drive);
# later runs ~3-6 min. Save the printed tables to the research log.
# ═══════════════════════════════════════════════════════════════════
import os, json, pickle, time, warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

RESEARCH_DIR   = "/app/outputs/research"
FEATURE_CACHE  = "/app/cache/t6_feature_rows.pkl"
HORIZONS       = [1, 3, 5]
MIN_TRAIN_DAYS = 20            # first walk-forward test week needs this much history
LOOKBACKS      = [7, 14, 21, 28, 60]
MARGINS        = [0.0, 0.03, 0.05]
RUN            = {"R1": True, "R2": True, "R3": True, "R4": True}
os.makedirs(RESEARCH_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FEATURE_CACHE), exist_ok=True)
TICKERS = list(SGX_STOCKS.keys())
CLASSES = ["bearish", "bullish", "neutral"]      # LabelEncoder order


def _q(sql, **p):
    with SUPABASE_ENGINE.connect() as conn:
        return conn.execute(text(sql), p).fetchall()


def _hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def _save(df, name):
    path = f"{RESEARCH_DIR}/{name}.csv"
    df.to_csv(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════
# DATA: complete dates, cached feature rows, daily closes
# ═══════════════════════════════════════════════════════════════════
_hdr("DATA")
ALL_DATES = [str(r[0]) for r in _q("""
    SELECT DISTINCT n.date FROM news_articles n
    WHERE n.category = 'company'
      AND EXTRACT(ISODOW FROM CAST(n.date AS DATE)) <= 5
      AND EXISTS (SELECT 1 FROM technical_signals t WHERE t.date = n.date AND t.ticker = n.ticker)
      AND EXISTS (SELECT 1 FROM macro_indicators m WHERE m.date = n.date)
      AND EXISTS (SELECT 1 FROM market_options o WHERE o.date = n.date)
      AND EXISTS (SELECT 1 FROM news_articles p WHERE p.ticker='POLITICAL' AND p.date = n.date)
    ORDER BY n.date""")]
print(f"  Complete T1-T4 dates: {ALL_DATES[0]} → {ALL_DATES[-1]} ({len(ALL_DATES)})")

# feature rows: cached per (ticker, date) so only new dates hit the DB
cache = {}
if os.path.exists(FEATURE_CACHE):
    with open(FEATURE_CACHE, "rb") as f:
        cache = pickle.load(f)
todo = [(t, d) for d in ALL_DATES for t in TICKERS if (t, d) not in cache]
print(f"  Feature cache: {len(cache)} rows cached, {len(todo)} to extract")
t0 = time.time()
for i, (t, d) in enumerate(todo, 1):
    try:
        cache[(t, d)] = extract_features_row(t, d)
    except Exception as e:
        cache[(t, d)] = None
    if i % 150 == 0:
        print(f"    {i}/{len(todo)}  ({time.time() - t0:.0f}s)")
        with open(FEATURE_CACHE, "wb") as f:
            pickle.dump(cache, f)
if todo:
    with open(FEATURE_CACHE, "wb") as f:
        pickle.dump(cache, f)

# daily closes → forward returns for every horizon (trading days)
import yfinance as yf
end = (datetime.strptime(ALL_DATES[-1], "%Y-%m-%d") + timedelta(days=14)).strftime("%Y-%m-%d")
CLOSES = {}
for t in TICKERS:
    h = yf.Ticker(t).history(start=ALL_DATES[0], end=end)["Close"].sort_index()
    h.index = [x.strftime("%Y-%m-%d") for x in h.index]
    CLOSES[t] = h
FWD = {}   # (ticker, date, h) -> % return
for t, s in CLOSES.items():
    idx, v = list(s.index), s.values
    for j, d in enumerate(idx):
        for hz in set(HORIZONS) | {1, 3}:
            if j + hz < len(idx):
                FWD[(t, d, hz)] = (v[j + hz] / v[j] - 1) * 100
print(f"  Prices: {sum(len(s) for s in CLOSES.values())} closes, "
      f"last {max(s.index[-1] for s in CLOSES.values())}")

POS = {d: i for i, d in enumerate(ALL_DATES)}


def build_frame(hz):
    rows = []
    for d in ALL_DATES:                       # chronological
        for t in TICKERS:
            fd = cache.get((t, d))
            r = FWD.get((t, d, hz))
            if fd is None or r is None:
                continue
            rows.append({**{c: fd.get(c, 0) for c in FEATURE_NAMES_T6},
                         "_t": t, "_d": d, "_r": r})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Walk-forward engine (shared by R1-R3)
# ═══════════════════════════════════════════════════════════════════
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _bases():
    b = {"LogReg": (LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"), True),
         "RandomF": (RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                            min_samples_split=5, class_weight="balanced",
                                            random_state=42, n_jobs=-1), False)}
    if HAS_XGB:
        b["XGBoost"] = (XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                      reg_lambda=1, objective="multi:softprob",
                                      eval_metric="mlogloss", random_state=42), False)
    return b


def _fit_leaky(base, X, y):
    """What production did until 19 Sep: calibrate on the training rows."""
    base.fit(X, y)
    try:
        from sklearn.frozen import FrozenEstimator            # sklearn >= 1.6
        return CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(X, y)
    except ImportError:
        return CalibratedClassifierCV(base, method="sigmoid", cv="prefit").fit(X, y)


def _fit_oof(base, X, y):
    """retrain_weekly v2: out-of-fold, time-ordered calibration."""
    return CalibratedClassifierCV(base, method="sigmoid",
                                  cv=TimeSeriesSplit(n_splits=4)).fit(X, y)


def _label(r, thr):
    return np.where(r > thr, "bullish", np.where(r < -thr, "bearish", "neutral"))


def walk_forward(hz, calib="oof"):
    """Weekly walk-forward. For each ISO week, train on rows whose label had
    matured before the week's first date (no look-ahead), predict the week.
    Returns one row per (model, ticker, date) with probabilities."""
    from sklearn.base import clone
    df = build_frame(hz)
    df["_week"] = pd.to_datetime(df["_d"]).dt.strftime("%G-W%V")
    weeks = sorted(df["_week"].unique())
    out = []
    for wk in weeks:
        test = df[df["_week"] == wk]
        t0 = POS[test["_d"].min()]
        train = df[df["_d"].map(POS) + hz < t0]            # label known before t0
        if train["_d"].nunique() < MIN_TRAIN_DAYS:
            continue
        thr = 0.5 * np.std(train["_r"])                     # train-only threshold
        ytr = _label(train["_r"].values, thr)
        if len(set(ytr)) < 3:
            continue
        le = LabelEncoder().fit(CLASSES)
        y = le.transform(ytr)
        Xtr, Xte = train[FEATURE_NAMES_T6].values, test[FEATURE_NAMES_T6].values
        sc = StandardScaler().fit(Xtr)
        # per-stock naive rule: that stock's majority class in its training rows
        naive = {t: Counter(ytr[train["_t"].values == t]).most_common(1)[0][0]
                 for t in TICKERS if (train["_t"].values == t).any()}
        yte = _label(test["_r"].values, thr)
        for name, (base, scaled) in _bases().items():
            fit = _fit_oof if calib == "oof" else _fit_leaky
            try:
                m = fit(clone(base), sc.transform(Xtr) if scaled else Xtr, y)
                P = m.predict_proba(sc.transform(Xte) if scaled else Xte)
            except Exception as e:
                print(f"    ⚠️  h={hz} {wk} {name} {calib}: {e!r}")
                continue
            full = np.zeros((len(Xte), 3))
            full[:, m.classes_] = P
            for k in range(len(test)):
                tk = test["_t"].iloc[k]
                out.append({"h": hz, "calib": calib, "model": name, "week": wk,
                            "ticker": tk, "date": test["_d"].iloc[k],
                            "pred": CLASSES[int(full[k].argmax())],
                            "p_max": float(full[k].max()),
                            "p_bear": full[k, 0], "p_bull": full[k, 1], "p_neu": full[k, 2],
                            "actual": yte[k], "naive": naive.get(tk, "neutral"),
                            "ret": test["_r"].iloc[k]})
    return pd.DataFrame(out)


def _metrics(g):
    y, p = g["actual"].values, g["pred"].values
    onehot = np.stack([(y == c).astype(float) for c in CLASSES], 1)
    probs = g[["p_bear", "p_bull", "p_neu"]].values
    hi = g[g["p_max"] >= 0.9]
    return pd.Series({
        "n":           len(g),
        "acc":         (y == p).mean(),
        "naive_acc":   (y == g["naive"].values).mean(),
        "edge":        (y == p).mean() - (y == g["naive"].values).mean(),
        "neutral_acc": (y == "neutral").mean(),        # always say 'neutral'
        "edge_vs_neu": (y == p).mean() - (y == "neutral").mean(),
        "macro_f1":    f1_score(y, p, average="macro", labels=CLASSES, zero_division=0),
        "brier":       ((probs - onehot) ** 2).sum(1).mean(),
        "mean_pmax":   g["p_max"].mean(),
        "share_ge90":  len(hi) / len(g),
        "hit_ge90":    (hi["actual"] == hi["pred"]).mean() if len(hi) else np.nan,
        "pred_bull%":  (p == "bullish").mean(),
        "pred_bear%":  (p == "bearish").mean(),
        "pred_neu%":   (p == "neutral").mean(),
        "act_bull%":   (y == "bullish").mean(),
        "act_bear%":   (y == "bearish").mean(),
    })


REUSE_WF = True    # True = if this session already ran the walk-forward, reuse it
if not (REUSE_WF and isinstance(globals().get("WF"), dict) and len(WF) == 2 * len(HORIZONS)):
    WF = {}
if not WF and (RUN["R2"] or RUN["R1"] or RUN["R3"]):
    _hdr("WALK-FORWARD (builds R1-R3 inputs)")
    for hz in HORIZONS:
        for cal in (["leaky", "oof"] if RUN["R2"] else ["oof"]):
            t0 = time.time()
            WF[(hz, cal)] = walk_forward(hz, cal)
            w = WF[(hz, cal)]
            print(f"  h={hz} {cal:<5}: {len(w)} predictions, "
                  f"{w['week'].nunique() if len(w) else 0} test weeks ({time.time() - t0:.0f}s)")
    pd.concat(WF.values()).to_csv(f"{RESEARCH_DIR}/walk_forward_predictions.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30,
              "display.float_format", lambda x: f"{x:.3f}")

# ═══════════════════════════════════════════════════════════════════
# R2 — horizon robustness × calibration
# ═══════════════════════════════════════════════════════════════════
if RUN["R2"]:
    _hdr("R2  HORIZON ROBUSTNESS × CALIBRATION (walk-forward, out-of-sample)")
    all_wf = pd.concat(WF.values())
    r2 = all_wf.groupby(["h", "calib", "model"]).apply(_metrics).reset_index()
    print(r2.to_string(index=False))
    _save(r2, "R2_horizon_calibration")
    print("\n  Read: edge = accuracy minus the per-stock naive rule (positive = useful).")
    print("  edge_vs_neu = accuracy minus 'always say neutral'. A model that beats the")
    print("  per-stock rule but not always-neutral has learned to say 'neutral', not skill.")
    print("  hit_ge90 should be close to 0.9 if probabilities are honest; share_ge90")
    print("  should drop sharply from 'leaky' to 'oof'. Brier: lower is better.")

    # per-stock edge at each horizon (oof, best model per horizon)
    oof = all_wf[all_wf["calib"] == "oof"]
    rows = []
    for hz in HORIZONS:
        g = oof[oof["h"] == hz]
        best = g.groupby("model").apply(lambda x: (x["actual"] == x["pred"]).mean()).idxmax()
        for t, gt in g[g["model"] == best].groupby("ticker"):
            rows.append({"h": hz, "model": best, "ticker": t, "n": len(gt),
                         "acc": (gt["actual"] == gt["pred"]).mean(),
                         "naive": (gt["actual"] == gt["naive"]).mean()})
    ps = pd.DataFrame(rows)
    ps["edge"] = ps["acc"] - ps["naive"]
    tab = {}
    for _, r in ps.iterrows():
        tab.setdefault(r["ticker"], {"ticker": r["ticker"]})[f"edge_h{r['h']}"] = r["edge"]
        tab[r["ticker"]][f"acc_h{r['h']}"] = r["acc"]
    print("\n  Per-stock accuracy and edge vs naive (best oof model per horizon):")
    print(pd.DataFrame(list(tab.values())).to_string(index=False))
    print(f"  Stocks beating naive: " +
          ", ".join(f"h={hz}: {(ps[ps.h == hz].edge > 0).sum()}/{ps[ps.h == hz].shape[0]}"
                    for hz in HORIZONS))
    _save(ps, "R2_per_stock_edge")

# ═══════════════════════════════════════════════════════════════════
# R1 — per-stock model consistency (cross-model agreement, target >= 80%)
# ═══════════════════════════════════════════════════════════════════
if RUN["R1"]:
    _hdr("R1  PER-STOCK MODEL CONSISTENCY (do the models agree? target >= 80%)")

    def agreement_table(df, label):
        models = sorted(df["model"].unique())
        rec = {}
        for tk, dt, m, pr, ac in df[["ticker", "date", "model", "pred", "actual"]].itertuples(index=False):
            r = rec.setdefault((tk, dt), {"ticker": tk, "date": dt, "actual": ac})
            r.setdefault(m, pr)
        wide = pd.DataFrame([r for r in rec.values() if all(m in r for m in models)])
        wide = wide.dropna(subset=["actual"])
        wide["agree"] = wide[models].nunique(axis=1) == 1
        champ = "RandomF" if "RandomF" in models else models[0]
        rows = []
        for t, g in wide.groupby("ticker"):
            a = g[g["agree"]]
            d = g[~g["agree"]]
            rows.append({"ticker": t, "n": len(g), "agree%": g["agree"].mean(),
                         "acc_when_agree": (a[champ] == a["actual"]).mean() if len(a) else np.nan,
                         "acc_when_split": (d[champ] == d["actual"]).mean() if len(d) else np.nan})
        tab = pd.DataFrame(rows)
        print(f"\n  [{label}]  models: {', '.join(models)}")
        print(tab.to_string(index=False))
        ok = tab[tab["agree%"] >= 0.8]["ticker"].tolist()
        print(f"  >= 80% agreement: {len(ok)}/{len(tab)}  {ok}")
        a_all = wide[wide["agree"]]
        print(f"  Overall: agree {wide['agree'].mean():.1%}; accuracy when all agree "
              f"{(a_all[champ] == a_all['actual']).mean():.1%} (n={len(a_all)}) vs "
              f"when they split {(wide[~wide['agree']][champ] == wide[~wide['agree']]['actual']).mean():.1%}"
              f" (n={(~wide['agree']).sum()})")
        return tab

    # (a) live stored predictions — what production actually said
    live = pd.DataFrame(_q("""
        SELECT DISTINCT ON (ticker, prediction_date, model_used)
               ticker, prediction_date::text, model_used, predicted_class, actual_class
        FROM task6_predictions
        WHERE actual_class IS NOT NULL
        ORDER BY ticker, prediction_date, model_used, id DESC"""),
        columns=["ticker", "date", "model", "pred", "actual"])
    if live["model"].nunique() >= 2:
        _save(agreement_table(live, "LIVE stored predictions, old calibration"), "R1_live")
    else:
        print("  Live table stores only the champion's rows — skipping the live view.")
    # (b) walk-forward h=3 with the fixed calibration
    if (3, "oof") in WF and len(WF[(3, "oof")]):
        _save(agreement_table(WF[(3, "oof")], "WALK-FORWARD h=3, out-of-fold calibration"),
              "R1_walkforward")
    print("\n  Read: if accuracy is clearly higher when the models agree, 'agreement' is a")
    print("  usable confidence gate for T7 (show the forecast only when models agree).")

# ═══════════════════════════════════════════════════════════════════
# R3 — champion selection: lookback × margin
# ═══════════════════════════════════════════════════════════════════
if RUN["R3"] and (3, "oof") in WF and len(WF[(3, "oof")]):
    _hdr("R3  RETRAIN WINDOW SENSITIVITY (champion replay, h=3, oof)")
    wf = WF[(3, "oof")].copy()
    models = sorted(wf["model"].unique())
    weeks = sorted(wf["week"].unique())
    wk_start = {w: wf[wf.week == w]["date"].min() for w in weeks}

    def replay(lookback, margin):
        champ, swaps, picked = "RandomF" if "RandomF" in models else models[0], 0, []
        for w in weeks[1:]:
            ds = wk_start[w]
            lo = (datetime.strptime(ds, "%Y-%m-%d") - timedelta(days=lookback)).strftime("%Y-%m-%d")
            # only predictions whose 3-day outcome had matured before this week
            hist = wf[(wf["date"] >= lo) & (wf["date"].map(POS) + 3 < POS[ds])]
            f1 = {m: f1_score(g["actual"], g["pred"], average="macro",
                              labels=CLASSES, zero_division=0)
                  for m, g in hist.groupby("model") if len(g) >= 30}
            if f1:
                best = max(f1, key=f1.get)
                if best != champ and f1[best] >= f1.get(champ, -1) + margin:
                    champ, swaps = best, swaps + 1
            picked.append(wf[(wf.week == w) & (wf.model == champ)])
        g = pd.concat(picked)
        return {"lookback": lookback, "margin": margin, "weeks": len(weeks) - 1,
                "swaps": swaps, "acc": (g.actual == g.pred).mean(),
                "macro_f1": f1_score(g.actual, g.pred, average="macro",
                                     labels=CLASSES, zero_division=0),
                "naive_acc": (g.actual == g.naive).mean()}

    r3 = pd.DataFrame([replay(L, M) for L in LOOKBACKS for M in MARGINS])
    fixed = {m: (wf[(wf.model == m) & (wf.week != weeks[0])].pipe(
                 lambda g: (g.actual == g.pred).mean())) for m in models}
    print(r3.to_string(index=False))
    print("  Never-swap reference (accuracy of each model held fixed): " +
          ", ".join(f"{m} {a:.3f}" for m, a in fixed.items()))
    _save(r3, "R3_lookback_margin")
    print("\n  Read: if no setting beats simply keeping one model, weekly champion")
    print("  switching is adding noise, not skill — choose the setting with fewest")
    print("  swaps among those within ~1 pt of the best accuracy.")

# ═══════════════════════════════════════════════════════════════════
# R4 — T5 evidence-layer replay: where does the bullish tilt come from?
# ═══════════════════════════════════════════════════════════════════
if RUN["R4"]:
    _hdr("R4  T5 REPLAY (rl_v7 audit features, re-scored under variants)")
    MARKET_WIDE = {"task1_political", "task3_macro", "task4_options"}
    rows = _q("""
        SELECT DISTINCT ON (ticker, analysis_date::date)
               ticker, analysis_date::date::text, features_result
        FROM llm_analyses_audit
        WHERE (features_result::jsonb)->>'_rule_layer_version' = 'rl_v7'
        ORDER BY ticker, analysis_date::date, id DESC""")
    rec = []
    for t, d, fr in rows:
        fr = fr if isinstance(fr, dict) else json.loads(fr)
        s_all = s_stock = s_mkt = 0.0
        for f in fr.get("features", []):
            w = float(f.get("baseline_weight", 0.0)) * (1 if f.get("direction") == "bullish" else -1)
            s_all += w
            if f.get("source") in MARKET_WIDE:
                s_mkt += w
            else:
                s_stock += w
        rec.append({"ticker": t, "date": d, "net": s_all, "stock": s_stock, "mkt": s_mkt,
                    "stored": (fr.get("score_breakdown") or {}).get("net_score"),
                    "r1": FWD.get((t, d, 1)), "r3": FWD.get((t, d, 3))})
    t5 = pd.DataFrame(rec)
    t5["demeaned"] = t5["net"] - t5.groupby("date")["net"].transform("mean")
    chk = (t5["stored"].astype(float) - t5["net"]).abs().max()
    print(f"  {len(t5)} rl_v7 readings, {t5.date.nunique()} dates. "
          f"Replay vs stored net score: max diff {chk:.4f} "
          f"({'OK' if chk < 0.01 else 'MISMATCH — check weights'})")
    print(f"  Mean net {t5.net.mean():+.3f} = stock-specific {t5.stock.mean():+.3f} "
          f"+ market-wide {t5.mkt.mean():+.3f}")

    thr1 = 0.5 * np.nanstd(t5["r1"].astype(float))

    def score(col, thr_score):
        s = t5[col]
        call = np.where(s > thr_score, "bullish", np.where(s < -thr_score, "bearish", "neutral"))
        g = t5.assign(call=call).dropna(subset=["r1"])
        act = _label(g["r1"].astype(float).values, thr1)
        dirn = g[g.call != "neutral"]
        hit = (np.sign(dirn["r1"].astype(float)) ==
               np.where(dirn.call == "bullish", 1, -1)).mean() if len(dirn) else np.nan
        # daily cross-sectional rank IC: does a higher score mean a better next day?
        ics = [x[col].rank().corr(x["r1"].astype(float).rank())
               for _, x in g.groupby("date") if x["r1"].notna().sum() >= 8]
        return {"variant": col, "threshold": thr_score,
                "bull%": (call == "bullish").mean(), "bear%": (call == "bearish").mean(),
                "class_hit_1d": (g["call"].values == act).mean(),
                "sign_hit_1d_when_called": hit,
                "mean_r1_bull": g[g.call == "bullish"]["r1"].astype(float).mean(),
                "mean_r1_bear": g[g.call == "bearish"]["r1"].astype(float).mean(),
                "rank_IC_1d": np.nanmean(ics), "IC_days": len(ics),
                "IC_t": np.nanmean(ics) / (np.nanstd(ics) / np.sqrt(len(ics))) if len(ics) > 2 else np.nan}

    variants = [("net", 0.35),                    # production rl_v7
                ("stock", 0.35),                  # drop market-wide rules
                ("demeaned", 0.35),               # remove the day's common tilt
                ("demeaned", 0.20),
                ("demeaned", float(np.nanpercentile(t5["demeaned"].abs(), 70)))]
    r4 = pd.DataFrame([score(c, th) for c, th in variants])
    print(r4.to_string(index=False))
    _save(r4, "R4_t5_replay")
    wk = t5.assign(week=pd.to_datetime(t5.date).dt.strftime("%G-W%V")).groupby("week").agg(
        mean_net=("net", "mean"), mean_mkt=("mkt", "mean"), mean_stock=("stock", "mean"),
        min_net=("net", "min"), n=("net", "size")).reset_index()
    print("\n  Weekly mean score split (shows the tilt is the market-wide part):")
    print(wk.to_string(index=False))
    _save(wk, "R4_weekly_split")
    print("\n  Read: rank_IC_1d is the investor-relevant number — does ranking the 15")
    print("  stocks by the score predict tomorrow's relative performance? |IC| > 0.05")
    print("  with IC_t > 2 would be meaningful; near 0 means the score describes, not")
    print("  predicts. 'demeaned' is the fix candidate for the zero-bearish problem.")

_hdr("DONE")
print(f"  CSVs saved in {RESEARCH_DIR}")
print("  Save everything printed above to the research log.")