"""Primary database writes and non-secret Task 1–4 settings.

The selected notebook cells depend on Setup's write helper. Keeping it here
prevents an absent helper from silently routing all writes into a fallback.
"""
import json
import re


WRITE_TABLES = {"technical_signals", "macro_indicators", "market_options"}


def load_settings(root, engine=None, source="files"):
    if source == "files":
        settings = {}
        for name in ("solution", "sgx", "system"):
            raw = (root / "settings" / f"{name}.json").read_text()
            settings[name] = json.loads(raw.replace("${project_root}", str(root)))
        return settings
    if source != "database":
        raise ValueError("runtime.settings_source must be files or database")
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT config_key, config_value FROM app_config "
                                 "WHERE config_key IN ('solution','sgx','system')")).fetchall()
    settings = dict(rows)
    missing = {"solution", "sgx", "system"} - set(settings)
    if missing:
        raise RuntimeError(f"Missing database settings: {sorted(missing)}")
    return {k: json.loads(v) if isinstance(v, str) else v for k, v in settings.items()}


def write_primary(engine, table_name, row):
    """Write one T2–T4 row to PostgreSQL; identifiers cannot come from input data."""
    from sqlalchemy import text
    if table_name not in WRITE_TABLES or not row:
        raise ValueError("Unsupported primary write table or empty row")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", key) for key in row):
        raise ValueError("Invalid database column name")
    columns = ", ".join(row)
    values = ", ".join(f":{key}" for key in row)
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO {table_name} ({columns}) VALUES ({values}) "
                          "ON CONFLICT DO NOTHING"), row)


def latest_complete_date(engine, tickers):
    """Require all covered technical rows plus shared inputs before T5 or T6."""
    from sqlalchemy import bindparam, text
    statement = text("""
        SELECT MAX(d) FROM (
          SELECT t.date AS d FROM technical_signals t
          WHERE t.ticker IN :tickers
            AND EXISTS (SELECT 1 FROM news_articles n
                        WHERE n.date=t.date AND n.category='company')
            AND EXISTS (SELECT 1 FROM news_articles n
                        WHERE n.date=t.date AND n.ticker='POLITICAL')
            AND EXISTS (SELECT 1 FROM macro_indicators m WHERE m.date=t.date)
            AND EXISTS (SELECT 1 FROM market_options o WHERE o.date=t.date)
          GROUP BY t.date HAVING COUNT(DISTINCT t.ticker)=:n
        ) complete
    """).bindparams(bindparam("tickers", expanding=True))
    with engine.connect() as conn:
        result = conn.execute(statement, {"tickers": list(tickers), "n": len(tickers)}).scalar()
    return str(result) if result is not None else None
