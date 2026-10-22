# ═══════════════════════════════════════════════════════════════════
# TASK 1 — News & Sentiment Analysis (Full Pipeline)
#
# Sources:
#   - Google News RSS  (per-stock query, ~50 articles/stock, cover ~164 outlets)
#   - Business Times   (filtered by stock name)
#   - Straits Times    (filtered by stock name)
#   - Political news   (12 configurable topics)
#
# Order:
#   Phase 1: Political news (shared, once per day)
#   Phase 2: Per stock — fetch → dedupe → filter → enrich → FinBERT
#            → aggregate → event detect → store
#
# Storage (⭐ CHANGED — Supabase primary):
#   Layer 1: Full raw JSON  → save_raw_article (Cell 11)
#                              Supabase raw_articles table primary,
#                              Drive JSON fallback if Supabase fails
#   Layer 2: news_articles → Supabase primary
#                              SQLite fallback if Supabase fails
# ═══════════════════════════════════════════════════════════════════

import feedparser
import requests
import os
import re
import sqlite3
import pandas as pd
from transformers import pipeline
from datetime import datetime
from html import unescape
from sqlalchemy import text

# ─────────────────────────────────────────────────────────
# PART A: ENRICHMENT MODULE (newspaper3k + sumy + decoder)
# ─────────────────────────────────────────────────────────

from googlenewsdecoder import gnewsdecoder
from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Sites that block scrapers
BLOCKED_DOMAINS = ["theedgesingapore.com"]


def extract_full_content(url, timeout=10):
    """Extract article body using newspaper3k. Returns (text, success)."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text_body = article.text.strip()
        if len(text_body.split()) >= 50:
            return text_body, True
        return "", False
    except Exception:
        return "", False


def summarize_text(text_body, max_sentences=5):
    """LexRank algorithmic summarization (no LLM, no token cost)."""
    if not text_body or len(text_body) < 500:
        return text_body
    try:
        parser     = PlaintextParser.from_string(text_body, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sentences  = summarizer(parser.document, max_sentences)
        summary    = " ".join(str(s) for s in sentences)
        return summary if summary else text_body[:500]
    except Exception:
        return text_body[:500]


def decode_url_if_needed(link):
    """Decode Google News URL → real URL. Pass-through if not Google News."""
    if not link:
        return None
    if "news.google.com" in link:
        try:
            result = gnewsdecoder(link, interval=1)
            if result.get("status"):
                return result["decoded_url"]
        except Exception:
            return None
        return None
    return link


def is_blocked_domain(url):
    """Check if URL is from a known blocked domain."""
    return any(d in url.lower() for d in BLOCKED_DOMAINS)


def strip_html(text_body):
    """
    Remove HTML tags and decode entities from text.

    Used when enrichment fails — RSS summaries from some sources contain
    <a>, <font>, &nbsp; etc. tags in the summary field.
    Handles three patterns:
      1. Complete tags:   <a href="x">text</a>  → 'text'
      2. Self-closing:    <br/>                  → ''
      3. Truncated tags:  <font color="#6f6f6   → '' (content was cut mid-tag

    This pollutes FinBERT input because the model sees HTML tags
    as content. Stripping yields clean text for sentiment analysis.
    """
    if not text_body:
        return ""

    # Remove complete HTML tags (closed with >)
    no_tags = re.sub(r'<[^>]+>', '', text_body)

    # Remove unclosed tags at end of string (truncated content)
    no_tags = re.sub(r'<[^>]*$', '', no_tags)

    # Decode HTML entities (&nbsp; &amp; &quot; etc)
    decoded = unescape(no_tags)

    # Collapse multiple whitespaces
    cleaned = re.sub(r'\s+', ' ', decoded).strip()

    return cleaned


def enrich_article(article):
    """
    Full enrichment pipeline:
    1. Decode Google News URL → real URL
    2. Skip if blocked domain
    3. Extract content with newspaper3k
    4. Summarize with LexRank if long

    Fallback behavior (when enrichment fails):
      Clean HTML from RSS summary to prevent polluting FinBERT input.
      This is critical for sources like The Edge Singapore whose RSS
      feeds contain <a>, <font>, &nbsp; etc. tags in the summary field.

    Adds: full_content, summary, content, enriched, real_url
    """
    link = article.get("link", "")

    # Clean any HTML from RSS-provided content upfront
    # (will be replaced with LexRank summary if enrichment succeeds)
    raw_content = article.get("content", "")
    cleaned_rss_content = strip_html(raw_content)

    real_url = decode_url_if_needed(link)
    if not real_url:
        article["full_content"] = ""
        article["summary"]      = article.get("title", "")
        article["content"]      = cleaned_rss_content   # Cleaned fallback
        article["enriched"]     = False
        article["real_url"]     = ""
        return article

    if is_blocked_domain(real_url):
        article["full_content"] = ""
        article["summary"]      = article.get("title", "")
        article["content"]      = cleaned_rss_content   # Cleaned fallback
        article["enriched"]     = False
        article["real_url"]     = real_url
        return article

    full_text, success = extract_full_content(real_url)

    if success:
        summary = summarize_text(full_text, max_sentences=5)
        article["full_content"] = full_text
        article["summary"]      = summary
        article["content"]      = summary
        article["enriched"]     = True
        article["real_url"]     = real_url
    else:
        article["full_content"] = ""
        article["summary"]      = article.get("title", "")
        article["content"]      = cleaned_rss_content   # Cleaned fallback
        article["enriched"]     = False
        article["real_url"]     = real_url

    return article


# ─────────────────────────────────────────────────────────
# PART B: NEWS PIPELINE (fetch → sentiment → store)
# ─────────────────────────────────────────────────────────

# ── Event detection ───────────────────────────────────
def detect_event(title):
    """Scan title for corporate event keywords."""
    t = title.lower()
    for event, keywords in EVENT_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return event
    return "none"


# ── Step 1: Fetch company news (Google News + BT + ST) ──
def fetch_newsapi(stock_name, ticker, date):
    """Fetch articles from 3 RSS sources for one stock."""
    articles = []

    # Google News (per-stock query)
    query = stock_name.replace(" ", "+") + "+SGX+Singapore"
    url   = f"https://news.google.com/rss/search?q={query}&hl=en-SG&gl=SG&ceid=SG:en"
    for e in feedparser.parse(url).entries:
        articles.append({
            "title":     e.get("title", ""),
            "source":    e.get("source", {}).get("title", "Google News"),
            "published": e.get("published", ""),
            "content":   e.get("summary", ""),
            "link":      e.get("link", ""),
            "category":  "company",
        })

    # Business Times (filtered)
    try:
        r = requests.get("https://www.businesstimes.com.sg/rss/companies-markets",
                         timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        for e in feedparser.parse(r.content).entries:
            t = e.get("title", "")
            if (stock_name.split()[0].lower() in t.lower() or
                ticker.replace(".SI", "") in t):
                articles.append({
                    "title":     t,
                    "source":    "Business Times",
                    "published": e.get("published", ""),
                    "content":   e.get("summary", ""),
                    "link":      e.get("link", ""),
                    "category":  "company",
                })
    except Exception:
        pass

    # Straits Times (filtered)
    try:
        r = requests.get("https://www.straitstimes.com/news/business/rss.xml",
                         timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        for e in feedparser.parse(r.content).entries:
            t = e.get("title", "")
            if (stock_name.split()[0].lower() in t.lower() or
                ticker.replace(".SI", "") in t):
                articles.append({
                    "title":     t,
                    "source":    "Straits Times",
                    "published": e.get("published", ""),
                    "content":   e.get("summary", ""),
                    "link":      e.get("link", ""),
                    "category":  "company",
                })
    except Exception:
        pass

    return articles


# ── Fetch political news (12 topics, shared) ──────────
def fetch_political_news():
    """Fetch global political news from configured topics."""
    political_articles = []
    pol_config         = CONFIG.get("political_news", {})
    topics             = pol_config.get("topics", [])
    articles_per_topic = pol_config.get("articles_per_topic", 4)

    for query in topics:
        q_encoded = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={q_encoded}&hl=en-SG&gl=SG&ceid=SG:en"
        for e in feedparser.parse(url).entries[:articles_per_topic]:
            political_articles.append({
                "title":     e.get("title", ""),
                "source":    e.get("source", {}).get("title", "Google News"),
                "published": e.get("published", ""),
                "content":   e.get("summary", ""),
                "link":      e.get("link", ""),
                "category":  "political",
                "topic":     query,
            })
    return political_articles


# ── Step 2: Deduplicate ───────────────────────────────
def deduplicate(articles):
    """Remove articles with near-identical titles."""
    seen, unique = set(), []
    for a in articles:
        key = a["title"].strip().lower()[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


# ── Step 3: Filter ────────────────────────────────────
def filter_articles(articles, stock_name, ticker=None):
    """
    Keep articles that genuinely reference this stock. Cap at MAX_ARTICLES.

    Matching strategy (in order of confidence):
      1. ticker code in title or content    (e.g. "D05" or "C38U")
      2. exact stock name (or first 3 words) in title or content
      3. distinctive first word + secondary word both present (AND)

    Why all three?
      - Some news uses ticker codes (analyst reports often do)
      - Some uses full name ("DBS Group Holdings")
      - Some uses just first word ("DBS reported...")

    Why AND for fallback (not OR)?
      The OLD logic used OR which produced false positives:
        - "Mapletree Industrial" matched any article with
          "industrial" in the title (e.g. "industrial sector outlook")
        - "CapitaLand Integrated" matched any article with
          "integrated" (e.g. "integrated circuits")
      AND requires BOTH words to be present → tighter match.

    Args:
      articles    : list of dicts with 'title' and 'content' fields
      stock_name  : full company name (e.g. "DBS Group")
      ticker      : optional ticker code (e.g. "D05.SI") for code-based matching
    """
    if not stock_name:
        return articles[:MAX_ARTICLES]

    name_lower = stock_name.lower()
    words      = stock_name.split()
    primary    = words[0].lower() if words else ""
    secondary  = words[1].lower() if len(words) > 1 else ""

    # Ticker code without exchange suffix (e.g. "D05" from "D05.SI")
    ticker_code = ""
    if ticker:
        ticker_code = ticker.replace(".SI", "").lower()

    filtered = []
    for a in articles:
        text_body = (a.get("title", "") + " " + a.get("content", "")).lower()

        # Strategy 1: Ticker code match (highest confidence)
        if ticker_code and ticker_code in text_body:
            filtered.append(a)
            continue

        # Strategy 2: Full name match (high confidence)
        if name_lower in text_body:
            filtered.append(a)
            continue

        # Strategy 3: Primary + secondary words BOTH present (medium confidence)
        if primary and secondary:
            if primary in text_body and secondary in text_body:
                filtered.append(a)
                continue
            # Strategy 3b: Very distinctive single-word fallback
            # If primary word is short (likely a ticker-like name e.g. "DBS"),
            # check if it appears with word boundaries (not inside another word).
            if len(primary) <= 4:
                if re.search(rf'\b{re.escape(primary)}\b', text_body):
                    filtered.append(a)
                    continue
        elif primary:
            # Single-word company names (e.g. "Singtel", "ComfortDelGro")
            if primary in text_body:
                filtered.append(a)
                continue

    return filtered[:MAX_ARTICLES]


# ── Step 4a: Load FinBERT ─────────────────────────────
def load_finbert():
    """Load ProsusAI/finbert once, reuse for all stocks."""
    print("  Loading FinBERT (~1 min first time)...")
    settings = CFG["ingestion"]["finbert"]
    model_path = os.environ.get("FINBERT_MODEL_DIR") or settings["model_id"]
    model = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path,
        truncation=True, max_length=settings["max_length"], device=-1,
    )
    print("  ✅ FinBERT ready")
    return model


# ── Step 4b: Classify ─────────────────────────────────
def classify_sentiment(articles, finbert_model):
    """
    Run FinBERT on title + summary content. Map to bullish/neutral/bearish.

    Input strategy:
      Combined `title + content` rather than title alone.
      - title:   article headline (often concise but can be misleading)
      - content: LexRank-summarised body (5 sentences, ~250 chars)

    Token limit handling:
      FinBERT inherits BERT-base's 512-token architecture limit. Pass
      truncation=True and max_length=512 so the tokenizer auto-truncates
      if needed. In practice, LexRank summaries (~250 chars) + title
      (~100 chars) = ~100 tokens, well below the 512 limit.

    Per-article fields added:
      sentiment        : "bullish" / "neutral" / "bearish"
      sentiment_score  : confidence score [0.0 - 1.0]
    """
    label_map = {
        "positive": "bullish",
        "negative": "bearish",
        "neutral":  "neutral"
    }

    for a in articles:
        try:
            title   = a.get("title", "")
            content = a.get("content", "")
            text_for_finbert = f"{title}. {content}"

            r = finbert_model(
                text_for_finbert,
                truncation=True,
                max_length=512
            )[0]
            a["sentiment"]       = label_map.get(r["label"].lower(), "neutral")
            a["sentiment_score"] = round(r["score"], 4)
        except Exception:
            a["sentiment"]       = "neutral"
            a["sentiment_score"] = 0.5

    return articles


# ── Step 4c: Aggregate per-stock sentiment ────────────
def aggregate_sentiment(articles):
    """
    Aggregate per-article sentiments into overall stock sentiment.

    Rules:
      bullish  if bull > 1.5x bear
      bearish  if bear > 1.5x bull
      neutral  otherwise

    Returns (overall_sentiment, average_signed_score) tuple.
    """
    if not articles:
        return "neutral", 0.0

    bull = sum(1 for a in articles if a.get("sentiment") == "bullish")
    bear = sum(1 for a in articles if a.get("sentiment") == "bearish")

    if bull > bear * 1.5:
        overall = "bullish"
    elif bear > bull * 1.5:
        overall = "bearish"
    else:
        overall = "neutral"

    # Signed average score for aggregation
    scores = []
    for a in articles:
        s = a.get("sentiment")
        score = a.get("sentiment_score", 0)
        if s == "bullish":
            scores.append(score)
        elif s == "bearish":
            scores.append(-score)
        else:
            scores.append(0)
    avg = sum(scores) / len(scores) if scores else 0.0
    return overall, round(avg, 4)


# ─────────────────────────────────────────────────────
# PART C: STORAGE (⭐ Supabase primary + SQLite fallback)
# ─────────────────────────────────────────────────────

def store_articles(ticker, date, articles, low_coverage):
    """
    Store per-stock news articles.
    Supabase primary, SQLite fallback.
    Also calls save_raw_article() for each (Cell 11 handles raw JSON).

    Returns count of articles written to news_articles.
    """
    if not articles:
        return 0

    # ── Layer 1: Save raw JSON (Cell 11 handles Supabase raw_articles) ──
    for a in articles:
        save_raw_article(a, ticker, date)

    # ── Prepare rows for news_articles table ──
    rows = []
    for a in articles:
        event_flag = detect_event(a.get("title", ""))
        rows.append({
            "ticker":          ticker,
            "date":            date,
            "title":           a["title"],
            "source":          a["source"],
            "published":       a["published"],
            "content":         a.get("content", ""),
            "sentiment":       a.get("sentiment"),
            "sentiment_score": a.get("sentiment_score"),
            "low_coverage":    1 if low_coverage else 0,
            "event_flag":      event_flag,
            "category":        "company",
        })

    df = pd.DataFrame(rows)

    # ── PRIMARY: Supabase ────────────────────────────
    try:
        df.to_sql(
            "news_articles",
            SUPABASE_ENGINE,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
        return len(rows)
    except Exception as e:
        print(f"    ⚠️  Supabase failed for {ticker}: {type(e).__name__}: {str(e)[:80]}")
        print(f"       Falling back to SQLite...")

    # ── FALLBACK: SQLite ─────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("news_articles", conn, if_exists="append", index=False)
        conn.close()
        print(f"       ⚠️  Written to SQLite fallback")
        return len(rows)
    except Exception as e:
        print(f"    ❌ SQLite ALSO failed for {ticker}: {e}")
        return 0


def store_political_news(date, articles, finbert_model):
    """
    Store political news. Enrichment + classification done here.
    Supabase primary, SQLite fallback.
    """
    if not articles:
        return 0

    # ── Enrich FIRST ──
    for a in articles:
        enrich_article(a)

    # ── Then classify (using LexRank content) ──
    classified = classify_sentiment(articles, finbert_model)

    # ── Save raw JSON (Cell 11 handles storage) ──
    for a in classified:
        save_raw_article(a, "POLITICAL", date)

    # ── Prepare rows for news_articles table ──
    rows = []
    for a in classified:
        rows.append({
            "ticker":          "POLITICAL",
            "date":            date,
            "title":           a["title"],
            "source":          a["source"],
            "published":       a["published"],
            "content":         a.get("content", ""),
            "sentiment":       a.get("sentiment"),
            "sentiment_score": a.get("sentiment_score"),
            "low_coverage":    0,
            "event_flag":      "none",
            "category":        "political",
        })

    df = pd.DataFrame(rows)

    # ── PRIMARY: Supabase ────────────────────────────
    try:
        df.to_sql(
            "news_articles",
            SUPABASE_ENGINE,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
        return len(rows)
    except Exception as e:
        print(f"    ⚠️  Supabase failed for POLITICAL: {type(e).__name__}: {str(e)[:80]}")

    # ── FALLBACK: SQLite ─────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("news_articles", conn, if_exists="append", index=False)
        conn.close()
        print(f"       ⚠️  Political written to SQLite fallback")
        return len(rows)
    except Exception as e:
        print(f"    ❌ SQLite ALSO failed: {e}")
        return 0


# ─────────────────────────────────────────────────────────
# PART D: MAIN RUNNER
# ─────────────────────────────────────────────────────────

def check_processed_count(date):
    """
    Count how many articles are already stored for given date.
    Supabase primary, SQLite fallback.
    Used to skip re-processing on second run of same date.
    """
    try:
        with SUPABASE_ENGINE.connect() as conn:
            return conn.execute(text("""
                SELECT COUNT(*) FROM news_articles WHERE date = :d
            """), {"d": date}).scalar()
    except Exception:
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT COUNT(*) FROM news_articles WHERE date=?", (date,)
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return 0


def run_task1(stocks, finbert_model, date=None):
    """
    Run complete Task 1 pipeline.

    Phase 1: Political news (once per day, shared across stocks)
    Phase 2: For each stock —
             fetch → dedupe → filter → enrich → FinBERT → aggregate → store
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    # ⭐ CHANGED: check via Supabase (with SQLite fallback)
    count = check_processed_count(date)
    if count > 0:
        print(f"  ⚠️  {date} already has {count} articles. Skipping.")
        return []

    print(f"\n[Task 1] News & Sentiment — {date}")
    print("=" * 70)
    print("Note: Content enrichment ~3s per article (decode + extract).\n")

    # ── Phase 1: Political news (shared, once per day) ──
    print("Phase 1: Political news (shared across stocks)")
    print("-" * 70)
    political   = fetch_political_news()
    pol_deduped = deduplicate(political)
    pol_stored  = store_political_news(date, pol_deduped, finbert_model)
    print(f"  📰 Political news stored: {pol_stored} articles\n")

    # ── Phase 2: Company news (per stock) ──────────────
    print("Phase 2: Company news (per stock)")
    print("-" * 70)
    results        = []
    total_enriched = 0

    for ticker, (name, sector) in stocks.items():
        # Step 1: Fetch
        articles     = fetch_newsapi(name, ticker, date)
        # Step 2: Deduplicate
        deduped      = deduplicate(articles)
        # Step 3: Filter by stock name
        filtered     = filter_articles(deduped, name, ticker)
        low_coverage = len(filtered) < MIN_ARTICLES_WARNING

        # ── Step 4: Enrich FIRST (decode + extract + summarize) ──
        # Critical: enrichment must happen BEFORE classify_sentiment,
        # so FinBERT sees LexRank-summarised content instead of RSS preview.
        enriched_articles = []
        enriched_count    = 0
        for a in filtered:
            a = enrich_article(a)
            enriched_articles.append(a)
            if a.get("enriched"):
                enriched_count += 1

        # ── Step 5: Classify with LexRank-enhanced content ──
        classified   = classify_sentiment(enriched_articles, finbert_model)
        overall, avg = aggregate_sentiment(classified)

        # ── Step 6: Store (enrichment already done) ──
        store_articles(ticker, date, classified, low_coverage)
        total_enriched += enriched_count

        icon   = "🟢" if overall == "bullish" else "🔴" if overall == "bearish" else "🟡"
        status = "⚠ LOW" if low_coverage else "OK"
        print(f"  {icon} {status:5s} {name:38s} | {len(classified):2d} arts | "
              f"enriched:{enriched_count:2d}/{len(classified):2d} | {overall:8s} ({avg:+.3f})")

        results.append({
            "ticker":    ticker,
            "name":      name,
            "sentiment": overall,
            "avg_score": avg,
            "enriched":  enriched_count
        })

    print(f"\n✅ Task 1 complete for {date}.")
    print(f"   Political news : {pol_stored} articles")
    print(f"   Company news   : {len(results)} stocks, {total_enriched} enriched")
    return results


print("✅ Task 1 (Full Pipeline with Enrichment) loaded.")
print("   Storage: Supabase primary + SQLite fallback")
print("   Usage:   finbert = load_finbert(); run_task1(SGX_STOCKS, finbert)")
