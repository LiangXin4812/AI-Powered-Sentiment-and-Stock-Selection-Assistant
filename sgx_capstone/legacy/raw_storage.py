# ═══════════════════════════════════════════════════════════════════
# RAW ARTICLE STORAGE
#
# Storage strategy: Supabase primary, Drive fallback
#   - 99% case:  Supabase raw_articles table receives write ✅
#   - 1% case:   Supabase down → fallback writes JSON to Drive
#
# Each article stores 13 fields including:
#   - Original metadata (title, source, link, published)
#   - full_content: newspaper3k extracted body
#   - real_url: post Google News URL decoder
#   - enriched: whether extraction succeeded
#   - category: 'company' or 'political'
#
# Design principle: Task 1 never loses data. If Supabase fails,
# Drive fallback triggers automatically and gets synced back later.
# ═══════════════════════════════════════════════════════════════════

import os
import json
import hashlib
from datetime import datetime
from sqlalchemy import text

# Drive path for fallback only (Supabase is primary)
RAW_DATA_ROOT = CONFIG["storage"]["raw_files_path"]


def ensure_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_article_id(article):
    """Unique 12-char ID from title (dedup key)."""
    title_clean = article.get("title", "").strip().lower()[:200]
    return hashlib.md5(title_clean.encode()).hexdigest()[:12]


def save_raw_article(article, ticker, date):
    """
    Save one raw article. Supabase primary; Drive fallback if that fails.

    Returns (article_id, saved_to_supabase) tuple.
      - article_id: unique ID (used as filename + dedup key)
      - saved_to_supabase: True if Supabase received the write,
                           False if Drive fallback was used instead.
    """
    article_id = get_article_id(article)

    # Build complete record
    record = {
        "article_id":   article_id,
        "ticker":       ticker,
        "date":         date,
        "fetched_at":   datetime.now().isoformat(),
        "title":        article.get("title", ""),
        "source":       article.get("source", ""),
        "published":    article.get("published", ""),
        "link":         article.get("link", ""),
        "content":      article.get("content", ""),
        "full_content": article.get("full_content", ""),
        "real_url":     article.get("real_url", ""),
        "enriched":     article.get("enriched", False),
        "category":     article.get("category", "company"),
    }

    # ── PRIMARY: Try Supabase first ──────────────────
    try:
        with SUPABASE_ENGINE.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw_articles (
                    article_id, ticker, date, title, url, source,
                    published, enriched, raw_data
                )
                VALUES (
                    :article_id, :ticker, :date, :title, :url, :source,
                    :published, :enriched, CAST(:raw_data AS JSONB)
                )
                ON CONFLICT (article_id, ticker, date) DO NOTHING
            """), {
                "article_id": article_id,
                "ticker":     ticker,
                "date":       date,
                "title":      record["title"],
                "url":        record["link"],
                "source":     record["source"],
                "published":  record["published"],
                "enriched":   record["enriched"],
                "raw_data":   json.dumps(record),
            })
        return article_id, True   # Supabase write succeeded

    except Exception as e:
        # ── FALLBACK: Supabase failed, write to Drive ─
        print(f"    ⚠️  Supabase raw_articles failed ({type(e).__name__}), "
              f"falling back to Drive for {article_id}")

        try:
            folder = f"{RAW_DATA_ROOT}/news/{date}/{ticker}"
            ensure_folder(folder)
            filepath = f"{folder}/{article_id}.json"

            if not os.path.exists(filepath):
                with open(filepath, "w") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)

            return article_id, False   # Fallback triggered
        except Exception as fallback_err:
            print(f"    ❌ Drive fallback ALSO failed: {fallback_err}")
            return article_id, False


def list_raw_articles(date=None, ticker=None):
    """
    Query raw articles from Supabase (fast, uses index).
    Replaces slow Drive os.walk.
    """
    where_parts = []
    params = {}
    if date:
        where_parts.append("date = :date")
        params["date"] = date
    if ticker:
        where_parts.append("ticker = :ticker")
        params["ticker"] = ticker
    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    sql = f"""
        SELECT article_id, ticker, date, raw_data
        FROM raw_articles
        {where_clause}
        ORDER BY date DESC, ticker, article_id
    """
    with SUPABASE_ENGINE.connect() as conn:
        return conn.execute(text(sql), params).fetchall()


def load_raw_article(article_id):
    """Load a single raw article by ID (Supabase)."""
    with SUPABASE_ENGINE.connect() as conn:
        row = conn.execute(
            text("SELECT raw_data FROM raw_articles WHERE article_id = :aid"),
            {"aid": article_id}
        ).scalar()
    return row  # already a dict (JSONB)


# Initialize (Drive fallback folder ready)
ensure_folder(f"{RAW_DATA_ROOT}/news")
print(f"✅ Raw article storage ready")
print(f"   Primary:  Supabase raw_articles table (JSONB)")
print(f"   Fallback: {RAW_DATA_ROOT}/news (only if Supabase fails)")

# Current state
with SUPABASE_ENGINE.connect() as conn:
    total = conn.execute(text("SELECT COUNT(*) FROM raw_articles")).scalar()
print(f"   Currently: {total} articles in Supabase")
