"""Explicit, repeatable entrypoints for the SGX capstone prototype.

Without the database: doctor, smoke, sample, figures, outlook-evidence,
research-report, followup. With the database: doctor --online, ingest, daily, report
(Research Report), signal-report (Daily Signal Report), run-all (the whole
day after the close: ingest -> daily -> signal-report -> reports/<date>/) and
check-dates (build and check the report for every stored date in a range) and
republish (the same, published to reports/<date>/ in the current format).

The database is the original external PostgreSQL database. This container
never creates a fresh data history. All network-backed commands fail clearly
when credentials or historical tables are missing.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from datetime import datetime

from .configuration import (ROOT, apply_config, audit_serving_artifact,
                            configure_report_modules, load_config, snapshot_config)
from .data_access import latest_complete_date, load_settings, write_primary

LEGACY = Path(__file__).parent / "legacy"


CREDENTIALS = ("SUPABASE_URL", "GROQ_API_KEY", "HF_TOKEN", "SMTP_HOST", "SMTP_PORT",
               "SMTP_USER", "SMTP_PASSWORD", "REPORT_TO")


def clean_credentials(env=os.environ) -> None:
    """Drop leading and trailing whitespace from credentials. A key saved with a
    trailing line break makes an invalid HTTP header: run #8 (31 Oct) collected
    Tasks 1-4 and then stopped at Task 5 with 'Invalid header value'."""
    for name in CREDENTIALS:
        if name in env and env[name] != env[name].strip():
            env[name] = env[name].strip()


def _execute(filename: str, ns: dict) -> None:
    path = LEGACY / f"{filename}.py"
    exec(compile(path.read_text(), str(path), "exec"), ns)


def _db_context(cfg, sha):
    url = os.environ.get("SUPABASE_URL", "")
    if not url:
        raise RuntimeError("SUPABASE_URL is required for live commands; see README.md")
    from sqlalchemy import create_engine, text
    engine = create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=2)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    settings = load_settings(ROOT, engine,
                             cfg.get("runtime", {}).get("settings_source", "files"))
    solution, market, system = (settings[k] for k in ("solution", "sgx", "system"))
    stocks = {ticker: (info["name"], info["sector"])
              for ticker, info in market["tickers"].items()}
    db_config = {
        "market": market["market"], "tickers": market["tickers"],
        "news": {**solution["news"], "sources": market["news_sources"]},
        "technical": solution["technical"],
        "macro": {"sora_fallback": market["macro_data"]["sora_fallback"],
                  "regime_thresholds": {
                      "high_rate_sora": solution["macro_regime"]["high_rate_threshold"],
                      "high_rate_cpi": solution["macro_regime"]["high_cpi_threshold"],
                      "low_rate_sora": solution["macro_regime"]["low_rate_threshold"],
                      "low_rate_cpi": solution["macro_regime"]["low_cpi_threshold"]}},
        "storage": system["storage"],
        "event_keywords": solution["event_keywords"],
        "api_keys": system["api_keys"],
        "political_news": market["political_news"],
        "options": solution["options"],
    }
    from sqlalchemy import text as _text
    import numpy as np
    import pandas as pd
    import yfinance as yf
    ns = dict(__name__="sgx_capstone.loaded", SUPABASE_ENGINE=engine,
              SUPABASE_URL=url, SGX_STOCKS=stocks, SOLUTION=solution,
              MARKET=market, SYSTEM=system, CONFIG=db_config, CFG=cfg,
              CONFIG_SHA=sha, text=_text, pd=pd, np=np, yf=yf,
              DB_PATH=str(ROOT / "data" / "fallback.sqlite"),
              MAX_ARTICLES=db_config["news"]["max_articles"],
              MIN_ARTICLES_WARNING=db_config["news"]["min_articles_warning"],
              LOOKBACK_DAYS=db_config["news"]["lookback_days"],
              NEWS_SOURCES=db_config["news"]["sources"],
              EVENT_KEYWORDS=db_config["event_keywords"],
              OPTIONS_CONFIG=db_config["options"],
              GROQ_API_KEY=os.environ.get("GROQ_API_KEY", ""))
    ROOT.joinpath("data").mkdir(exist_ok=True)
    apply_config(cfg, ns)
    ns["mirror_to_supabase"] = lambda table, row: write_primary(engine, table, row)
    ns["apply_config"] = lambda settings, target=None: apply_config(settings, ns if target is None else target)
    return ns


def doctor(cfg, sha, online=False):
    print(f"Configuration {cfg['config_version']} · SHA {sha}")
    print(f"Task 5: {cfg['t5']['model_version']} / {cfg['t5']['rule_layer_version']}")
    print(f"Task 6: {cfg['t6']['expected_forward_horizon']} sessions")
    if not (ROOT / "outputs" / "research" / "R11_summary.csv").exists():
        raise FileNotFoundError("The supplied R11 research results are missing")
    print("Required config, prompts, models and research results: present")
    load_settings(ROOT, source=cfg["runtime"]["settings_source"]) if cfg["runtime"]["settings_source"] == "files" else None
    audit = audit_serving_artifact(cfg)
    print(f"Bundled T6 artifact: {audit}")
    if not audit["passes_metadata_preflight"]:
        raise RuntimeError("Bundled T6 artifact failed metadata preflight")
    if online:
        ns = _db_context(cfg, sha)
        print(f"PostgreSQL data source: connected · {len(ns['SGX_STOCKS'])} tickers")
        from sqlalchemy import inspect
        required = {"news_articles": {"category", "event_flag", "published"},
                    "raw_articles": {"raw_data", "url"},
                    "technical_signals": {"weekly_signal"},
                    "macro_indicators": {"vix", "us_yield_spread"},
                    "market_options": {"pc_volume_ratio", "options_signal"}}
        schema = inspect(ns["SUPABASE_ENGINE"])
        for table, columns in required.items():
            present = {c["name"] for c in schema.get_columns(table)}
            if columns - present:
                raise RuntimeError(f"{table} missing required columns: {sorted(columns-present)}")
        print("Required primary input tables and columns: present")
    else:
        print("Database and external feeds: not checked; run doctor --online with credentials")


def ingest(ns, date):
    today = datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat()
    if date != today:
        from .automation import report_date
        if date != report_date():
            raise ValueError("Live news, macro and options feeds cannot reconstruct "
                             "a historical date. Ingest only today's Singapore date, or "
                             "the latest closed SGX session before the next one opens; "
                             "use report --date to read stored historical evidence.")
    for file in ("raw_storage", "task1_news", "task2_technical", "task3_macro", "task4_options"):
        _execute(file, ns)
    import nltk
    for resource in ("punkt", "punkt_tab"):
        nltk.download(resource, quiet=True)
    model = ns["load_finbert"]()
    ns["run_task1"](ns["SGX_STOCKS"], model, date=date)
    ns["run_task2"](ns["SGX_STOCKS"], date=date)
    ns["run_task3"](date=date)
    ns["run_task4"](ns["SGX_STOCKS"], date=date)
    latest = latest_complete_date(ns["SUPABASE_ENGINE"], ns["SGX_STOCKS"])
    if latest != date:
        raise RuntimeError(f"Tasks 1–4 are incomplete in the primary database on "
                           f"{date}; latest complete date is {latest}. Check feed "
                           "and write failures before forecasting.")
    print(f"Tasks 1–4 primary database inputs complete for {date}")


def daily(ns, cfg):
    for file in ("task6_features", "task5_runtime", "task5_llm_config", "task5_daily", "task6_runtime"):
        _execute(file, ns)
        apply_config(cfg, ns)
    complete = lambda: latest_complete_date(ns["SUPABASE_ENGINE"], ns["SGX_STOCKS"])
    ns["_t5_latest_complete_date"] = complete
    ns["get_latest_complete_data_date"] = complete
    if complete() is None:
        raise RuntimeError("No date has all required primary T1–T4 inputs")
    if not ns["t6_preflight"](verbose=False):
        raise RuntimeError("Task 6 model preflight failed; do not serve its predictions")
    ns["run_task5_daily"](pause_between=cfg["llm"]["backoff_base_s"] * 2)
    data_date = ns["get_latest_complete_data_date"]()
    from sqlalchemy import text
    with ns["SUPABASE_ENGINE"].connect() as conn:
        n5 = conn.execute(text("SELECT COUNT(DISTINCT ticker) FROM llm_analyses "
                               "WHERE analysis_date::date = CAST(:d AS DATE)"),
                          {"d": data_date}).scalar()
    if n5 < len(ns["SGX_STOCKS"]):
        raise RuntimeError(f"Only {n5}/{len(ns['SGX_STOCKS'])} T5 readings on "
                           f"{data_date}; rerun daily after restoring data or LLM access")
    ns["run_task6_daily"]()
    with ns["SUPABASE_ENGINE"].connect() as conn:
        n6 = conn.execute(text("SELECT COUNT(DISTINCT ticker) FROM task6_predictions "
                               "WHERE prediction_date = CAST(:d AS DATE) "
                               "AND is_champion = TRUE"), {"d": data_date}).scalar()
    if n6 < len(ns["SGX_STOCKS"]):
        raise RuntimeError(f"Only {n6}/{len(ns['SGX_STOCKS'])} champion "
                           f"predictions on {data_date}; report generation stopped")


def report(ns, cfg, date=None):
    import task7_report, task7_generate
    configure_report_modules(cfg)
    _execute("task5_llm_config", ns) if "build_llm_prompt" in ns else None
    if "install_t7_llm_layer" in ns:
        ns["install_t7_llm_layer"](verbose=False)
    # The report builder imports the already configured modules. No reload.
    path = Path(__file__).parent / "report_builder.py"
    exec(compile(path.read_text(), str(path), "exec"), ns)
    ns["PRODUCTION_AUDIT"] = audit_serving_artifact(cfg)
    ns["NET_SCORE_LINE"] = cfg["t5"]["net_score_threshold"]
    ns["OUT_DIR"] = str(ROOT / "outputs" / "report")
    ns["RESEARCH_DIR"] = str(ROOT / "outputs" / "research")
    if date:
        ns["REPORT_DATE"] = date
    data = ns["gather"]()
    if not data.get("report_date") or not data.get("day"):
        raise RuntimeError("No complete stored T5/T6 report date; run doctor --online "
                           "and verify the historical task tables")
    stats = ns["analyse"](data)
    stats["research"] = ns["load_research"](data)
    focus = ns["pick_focus"](data, stats)
    output = ns["build_doc"](data, stats, focus)
    print(f"Report: {output}")
    return output


def signal_report(ns, cfg, date=None, sessions=None, save_snapshot=False, use_llm=True):
    """Daily Signal Report for a stored date (live database). --sessions sets
    the period (the 'time frame'): the last N trading days to the report date."""
    if sessions:
        cfg = {**cfg, "t7": {**cfg["t7"], "period_sessions": int(sessions)}}
    import task7_report, task7_generate                          # noqa: F401
    configure_report_modules(cfg)
    for file in ("task6_features", "task6_runtime"):
        _execute(file, ns)
        apply_config(cfg, ns)
    from .daily_report import build_live
    key = os.environ.get("GROQ_API_KEY") if use_llm else None
    out = build_live(ns, cfg, date, save_snapshot=save_snapshot, api_key=key)
    print(f"Daily Signal Report: {out}")
    return out


def run_all(ns, cfg, date=None, use_llm=True, force=False):
    """The whole trading day, unattended: Tasks 1-4 for the latest closed SGX
    session, then Tasks 5 and 6, then the Daily Signal Report, published to
    reports/<date>/ with an index. Exits quietly when that session's report is
    already published (a weekend, an SGX holiday, or a second scheduled run)."""
    from .automation import already_published, email, is_trading_day, publish, report_date
    if date is None and not force:
        date = report_date()
        if date is None:
            print("An SGX session is in progress; run after 17:30 Singapore time.")
            return None
        if already_published(ROOT, date):
            print(f"{date}: the report for the latest closed SGX session is already published; nothing to do.")
            return None
        print(f"Report date: {date} (latest closed SGX session)")
    else:
        date = date or datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat()
        if not force and not is_trading_day(date):
            print(f"{date}: no SGX trading session (weekend or public holiday); nothing to do.")
            return None
    ingest(ns, date)
    daily(ns, cfg)
    out = signal_report(ns, cfg, date, use_llm=use_llm)
    files = publish(ROOT, Path(out), date)
    print("Published: " + ", ".join(str(v) for v in files.values()))
    if email(files, date):
        print(f"E-mailed to {os.environ.get('REPORT_TO')}")
    return files


def check_dates_cmd(ns, cfg, start, end, use_llm=False, publish_all=False):
    import task7_report, task7_generate                          # noqa: F401
    configure_report_modules(cfg)
    for file in ("task6_features", "task6_runtime"):
        _execute(file, ns)
        apply_config(cfg, ns)
    from .automation import check_dates
    key = os.environ.get("GROQ_API_KEY") if use_llm else None
    return check_dates(ns, cfg, start, end, ROOT, api_key=key, publish_all=publish_all)


def sample(cfg, sessions=None, date=None, use_llm=False):
    """Offline Daily Signal Report: 21 Sep 2026 by default (the saved copy of
    the live database, AI text included), --date 2026-09-17 or 2026-09-18 for
    the stored-records version. --sessions sets the period in trading days."""
    import task7_report, task7_generate                          # noqa: F401
    configure_report_modules(cfg)
    from .daily_report import build_sample
    key = os.environ.get("GROQ_API_KEY") if use_llm else None
    if use_llm and not key:
        print("  note: --llm needs GROQ_API_KEY; using the template text")
    out = build_sample(period_sessions=sessions or cfg["t7"]["period_sessions"], date=date,
                       api_key=key, llm_settings=cfg.get("llm"))
    print(f"Sample Daily Signal Report: {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor").add_argument("--online", action="store_true")
    sub.add_parser("daily")
    sub.add_parser("smoke")
    sub.add_parser("research-report")
    sub.add_parser("outlook-evidence")
    sub.add_parser("figures")
    sub.add_parser("followup", help="R15: what followed the report (flags vs next moves, record by call type)")
    sample_cmd = sub.add_parser("sample")
    sample_cmd.add_argument("--sessions", type=int)
    sample_cmd.add_argument("--date", help="2026-09-21 (default), 2026-09-17, 2026-09-18, or a date saved with --save-snapshot")
    sample_cmd.add_argument("--llm", action="store_true",
                            help="let the Task 5 model write the summary lines (needs GROQ_API_KEY)")
    signal_cmd = sub.add_parser("signal-report")
    signal_cmd.add_argument("--date")
    signal_cmd.add_argument("--sessions", type=int, help="period length in trading days (default 5)")
    signal_cmd.add_argument("--save-snapshot", action="store_true",
                            help="also save the inputs to tests/fixtures so 'sample --date' rebuilds it offline")
    signal_cmd.add_argument("--no-llm", action="store_true", help="template text only, even with GROQ_API_KEY")
    run_cmd = sub.add_parser("run-all", help="after the close: Tasks 1-7 for the latest closed session and publish the report")
    run_cmd.add_argument("--date", help="default: the latest closed SGX session (skipped if already published)")
    run_cmd.add_argument("--no-llm", action="store_true")
    run_cmd.add_argument("--force", action="store_true", help="run even if no SGX bar is found for the date")
    check_cmd = sub.add_parser("check-dates", help="build and check the report for every stored date in a range")
    check_cmd.add_argument("--start", default="2026-08-17")
    check_cmd.add_argument("--end", default=datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat())
    check_cmd.add_argument("--llm", action="store_true", help="also let the AI write the text (slower)")
    rep_cmd = sub.add_parser("republish", help="rebuild every stored date's report in the current format "
                                               "and publish it to reports/<date>/")
    rep_cmd.add_argument("--start", default="2026-08-17")
    rep_cmd.add_argument("--end", default=datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat())
    rep_cmd.add_argument("--no-llm", action="store_true", help="template text only (no AI call)")
    ingest_cmd = sub.add_parser("ingest")
    ingest_cmd.add_argument("--date", default=datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat())
    report_cmd = sub.add_parser("report")
    report_cmd.add_argument("--date")
    args = parser.parse_args(argv)
    clean_credentials()
    cfg, sha = load_config()
    sys.path.insert(0, str(ROOT / "src"))
    if args.command == "research-report":
        from .research.r14_outlook import run as run_r14
        run_r14(ROOT)
        from .research_report import build
        print(f"Frozen research review: {build(ROOT)}")
        return
    if args.command == "outlook-evidence":
        from .research.r14_outlook import run as run_r14
        ev = run_r14(ROOT)
        q = ev["quiet"]
        print(f"R14 ({ev['event_definition']}): range held {ev['range']['range_coverage']:.1%}; "
              f"quiet five moved {q['quiet_move_rate']:.1%} vs {q['rest_move_rate']:.1%} "
              f"(lower on {q['dates_quiet_strictly_lower']}/{q['dates']} dates, "
              f"{q['dates_tied']} tied); direction AUC {ev['direction']['auc_among_moves']:.3f}")
        return
    if args.command == "followup":
        from .research.r15_report_followup import run as run_r15, summary
        print(summary(run_r15(ROOT)))
        return
    if args.command == "figures":
        from .figures import build as build_figures
        for path in build_figures(ROOT):
            print(path)
        return
    if args.command == "sample":
        return sample(cfg, args.sessions, args.date, args.llm)
    if args.command == "smoke":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    if args.command == "doctor":
        return doctor(cfg, sha, args.online)
    snapshot_config(sha)
    ns = _db_context(cfg, sha)
    if args.command == "ingest":
        return ingest(ns, args.date)
    if args.command == "daily":
        return daily(ns, cfg)
    if args.command == "signal-report":
        return signal_report(ns, cfg, args.date, args.sessions, args.save_snapshot, not args.no_llm)
    if args.command == "run-all":
        return run_all(ns, cfg, args.date, not args.no_llm, args.force)
    if args.command == "check-dates":
        return check_dates_cmd(ns, cfg, args.start, args.end, args.llm)
    if args.command == "republish":
        return check_dates_cmd(ns, cfg, args.start, args.end, not args.no_llm, publish_all=True)
    return report(ns, cfg, args.date)


if __name__ == "__main__":
    main()
