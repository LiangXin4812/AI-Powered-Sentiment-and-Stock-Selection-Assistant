"""Task 7 narrative layer for the Daily Signal Report (19 Oct 2026).

The same language model as Task 5 (config llm.model, served by Groq) writes
the 30-second summary and one 'In short' line per stock card. It chooses the
words; code supplies every number. The model sees each figure only as a named
placeholder ({top_period_move}) and must not write a digit itself, so a number
can never be invented or attached to the wrong claim (the limit found for the
outlook explanation; final report, Section 7.6). Checks before use:

  N1  valid JSON with the expected blocks
  N2  no advice, price targets or certainty (Task 7 rules R1-R3)
  N3  no digit outside a placeholder; no unknown placeholder
  N4  a card uses only its own placeholders
  N5  a rise word is never next to a negative figure, nor a fall word next to
      a positive one
  N6  no 'bullish' / 'bearish' (the report's words are positive / negative)
  N7  length limits
  N8  today's move is not compared with the 5-day move
  N9  the news is not said to explain a move the report says it does not
  N10 Task 6 is never given more than little weight
  N11 a summary line keeps to the facts of its code-written draft line: no
      placeholder or stock name the draft line lacks, none of its names dropped

The summary and the cards are asked for in two separate calls. A block that
fails a check is sent back once with the failed checks listed (one retry); a
block that still fails is replaced by the template text, and the flags
are printed on the reviewer page. Without an API key the template is used.
"""
from __future__ import annotations

import json
import re

PROMPT = """You write two short parts of a daily report on 15 Singapore stocks for a beginner investor.
The report's tables already show every number. Your job is plain-English sentences.

How the system works (use this framing):
- The news-and-signals reading (internally "Task 5") reads today's news and price-chart indicators and
  says what they add on top of the price itself
  (a reading: positive, neutral or negative). It describes today; it is not a forecast.
- The forecast model (internally "Task 6") forecasts the next 3 trading days. It is weak overall; its
  record on each stock is shown
  against a simple rule, and calls that trail the rule should be treated as weak.

STRICT RULES
1. Never write a digit (write "three", "five"; "Task 5" and "Task 6" are allowed). Every figure, percentage
   or count from the data must be one of the placeholders listed below, written exactly with braces,
   e.g. {{top_period_move}}. Do not invent placeholders, and never spell a figure out in words either
   (no "a fifty-percent chance"): the only way to state a figure is its placeholder.
2. No advice (buy, sell, hold, recommend, you should), no price targets, no certainty about future prices
   (never "will rise" / "will fall"). Do not use the words bullish or bearish.
3. A card may use only its own placeholders (c1_ for card c1, and so on) plus the general ones.
   The summary is about all 15 stocks and uses only the general placeholders (none starting with c1_, c2_, c3_).
4. Summary (the day's market read): rewrite the DRAFT lines below in plain, friendly English for a beginner,
   as a JSON list with one string per draft line, in the same order. Keep the facts, stock names and
   placeholders of each line in that line; add no fact, stock name, count or comparison the line does not
   already contain. At most 55 words each.
5. Each card: 2 or 3 sentences, at most 70 words: what happened, what today's news and price say together
   and the conclusion about the last five days, what Task 6 expects and how much weight that deserves.
   Write for a reader: never name the facts themselves (no "five-day context", "news explains", "task6_record").
6. Use each "news_explains_the_move" fact exactly. If it is "no", "not clearly" or "not enough news",
   say plainly that the news does not explain the move. Never say the news or the reading explains,
   drives or accounts for a move unless that fact is "same direction". Use "today_news_vs_price" the
   same way: never say the price has already moved or reflected the news when that fact says not yet,
   nor the reverse.
7. Do not compare today's move with the 5-day move (no "matching", "same as", "in line with").
   Write dates only as placeholders ({{c1_big_day}}) and the period as "over {{period}}".
   Put a figure straight after its verb: "fell {{c1_period_move}} over {{period}}".
8. Task 6 deserves little weight on every stock, even where its record is about as good as a simple rule:
   never call its forecast moderate, strong, reliable or trustworthy.

9. The reader does not know the internal names: write "the news-and-signals reading" and "the forecast
   model", never "Task 5" or "Task 6".

{draft_block}

PLACEHOLDERS (value shown for your understanding only; write the placeholder, never the value)
{facts}

{return_line}"""

RISE = r"(?:rose|rise|risen|gained|gain|climbed|jumped|up|higher|rallied|advanced|surged|rebounded)"
FALL = r"(?:fell|fall|fallen|dropped|drop|declined|slid|down|lower|lost|slipped|sank|eased|tumbled|retreated)"
_PH = re.compile(r"\{([a-z0-9_]+)\}")
# Design constants, not data: the 15 stocks, the 3- and 5-day horizons, Task 1-7.
_ALLOWED = re.compile(r"\btask [1-7]\b|\b15 (?:singapore |sgx[- ]listed |sgx )?(?:stocks|shares|companies)\b"
                      r"|\b[35][- ](?:trading[- ])?days?\b|\b[35]-session\b", re.I)


def _short(name):
    from .daily_report import short
    return short(name)


def build_facts(D):
    """Placeholder -> (display string, meaning). Built from the same values as
    the template text in sections 1-3."""
    from .daily_report import (pct, largest_day, news_tone, pick_cards, FORECAST_WORD, EVIDENCE_WORD,
                               PATTERN_NAME, PATTERN_WORD, _decisive, _calm, news_count)
    rows = list(D["rows"].values())
    import numpy as np
    avg = float(np.nanmean([r["move"] for r in rows]))
    up, dn = sum(r["move"] >= 0.05 for r in rows), sum(r["move"] <= -0.05 for r in rows)
    top_t = max(rows, key=lambda r: abs(r["move"]))
    big = max(rows, key=lambda r: abs(r["period_ret"]))
    looks = sorted(D["outlook"].values(), key=lambda o: o["range"]["half_width_pct"])
    pos = [_short(r["name"]) for r in rows if r["sentiment"] == "bullish"]
    neg = [_short(r["name"]) for r in rows if r["sentiment"] == "bearish"]
    ld = largest_day(big)
    f = {
        "date": (f"{__import__('pandas').Timestamp(D['date']):%a %d %b}", "report date"),
        "period": (f"{__import__('pandas').Timestamp(D['period']['first']):%d}–"
                   f"{__import__('pandas').Timestamp(D['period']['end']):%d %b}", "the last 5 trading days"),
        "avg_today": (pct(avg), "average change of the 15 stocks today"),
        "n_up": (str(up), "stocks that rose today"), "n_down": (str(dn), "stocks that fell today"),
        "n_flat": (str(15 - up - dn), "stocks unchanged today"),
        "top_today_move": (pct(top_t["move"]), f"today's biggest move, by {top_t['name']}"),
        "today_biggest_mover": (top_t["name"], "the stock with today's biggest move"),
        "five_day_biggest_mover": (big["name"], "the stock with the biggest move over the last 5 days"),
        "n_positive": (str(len(pos)), "stocks with a positive Task 5 reading today"),
        "n_negative": (str(len(neg)), "stocks with a negative Task 5 reading today"),
        "top_period_move": (pct(big["period_ret"]), f"biggest move over the last 5 days, by {big['name']}"),
        "top_period_day_move": (pct(ld[1]) if ld else "—", f"{big['name']}'s biggest single day"),
        "top_period_day": (f"{__import__('pandas').Timestamp(ld[0]):%a %d %b}" if ld else "—",
                           f"date of {big['name']}'s biggest single day"),
        "n_fc_none": (str(sum(r["t6_class"] == "neutral" for r in rows)), "Task 6 forecasts of no big move"),
        "n_fc_rise": (str(sum(r["t6_class"] == "bullish" for r in rows)), "Task 6 forecasts of a rise"),
        "n_fc_fall": (str(sum(r["t6_class"] == "bearish" for r in rows)), "Task 6 forecasts of a fall"),
        "n_trusted": (str(sum(not r["withheld"] for r in rows)),
                      "stocks where Task 6's record is about as good as a simple rule (✓)"),
        "swing_low": (f"±{looks[0]['range']['half_width_pct']:.1f}%", f"smallest normal 3-day swing ({looks[0]['name']})"),
        "swing_high": (f"±{looks[-1]['range']['half_width_pct']:.1f}%", f"largest normal 3-day swing ({looks[-1]['name']})"),
        "avg_period": (pct(float(np.nanmean([r["period_ret"] for r in rows]))),
                       "average change of the 15 stocks over the last 5 days"),
        "vix": (f"{float((D.get('inputs_today') or {}).get('vix') or 0):.1f}", "US volatility index (VIX) today"),
        "pc_ratio": (f"{float((D.get('inputs_today') or {}).get('pc_ratio') or 0):.2f}",
                     "put/call volume ratio of the Singapore ETF options today"),
    }
    secs = _sectors(D)
    for i, (sec, v) in enumerate(secs[:3], 1):
        f[f"sw{i}"] = (pct(v), f"{sec}: average change over the last 5 days (weakest sectors)")
    for i, (sec, v) in enumerate(secs[-2:][::-1], 1):
        f[f"ss{i}"] = (pct(v), f"{sec}: average change over the last 5 days (strongest sectors)")
    for i, r in enumerate(_movers(D), 1):
        f[f"mv{i}"] = (pct(r["period_ret"]), f"{r['name']}: change over the last 5 days (biggest movers)")
    context = {"positive_readings": pos or ["none"], "negative_readings": neg or ["none"],
               "five_day_biggest_mover_news": news_tone(big),
               "five_day_biggest_mover_news_explains_the_move": explain_verdict(big),
               "calmer_than_most": [_short(o["name"]) for o in _calm(D)]}
    cards = {}
    for k, r in enumerate(pick_cards(D), 1):
        c = f"c{k}"
        ldr = largest_day(r)
        f[f"{c}_period_move"] = (pct(r["period_ret"]), f"{r['name']}'s change over the last 5 days")
        f[f"{c}_today_move"] = (pct(r["move"]), f"{r['name']}'s change today")
        f[f"{c}_big_day_move"] = (pct(ldr[1]) if ldr else "—", f"{r['name']}'s biggest single day")
        f[f"{c}_big_day"] = (f"{__import__('pandas').Timestamp(ldr[0]):%a %d %b}" if ldr else "—",
                             f"date of {r['name']}'s biggest single day")
        f[f"{c}_n_articles"] = (str(news_count(r)), f"company articles collected today on {r['name']}")
        p = r.get("t6_prob")
        f[f"{c}_chance"] = ("—" if p is None else (">99%" if p > 0.995 else f"{p * 100:.0f}%"),
                            f"Task 6's stated chance for its forecast on {r['name']}")
        from .daily_report import context as _context, evidence_vs_price as _evp
        cards[c] = {"stock": r["name"], "five_day_direction": "fell" if r["period_ret"] < 0 else "rose",
                    "today_news_vs_price": PLAIN_PRICE.get(_evp(r["sentiment"], r.get("same_day")),
                                                           "no news signal today"),
                    "five_day_context": _context(r, D)["conclusion"],
                    "news_explains_the_move": explain_verdict(r), "news_today": __import__("sgx_capstone.daily_report", fromlist=["tone_label"]).tone_label(r).lower(), "task5_reading": EVIDENCE_WORD[r["sentiment"]].lower(),
                    "task5_decided_by": _decisive(r) or "no single rule",
                    "task6_forecast": FORECAST_WORD[r["t6_class"]].lower(),
                    "task6_record": "worse than a simple rule on this stock, so it deserves little weight" if r["withheld"]
                                    else "about as good as a simple rule on this stock, still only a weak signal"}
    return f, context, cards


MARKET_READ_LEADS = ("Today. ", "Why. ", "Backdrop. ", "New today. ", "Next 3 days. ",
                     "For you. ")
MARKET_READ_REFS = ("", " See section 2.", "", " See section 2.", " See section 3.", "")
NUM_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen")   # the draft carries no digit outside a placeholder
PLAIN_PRICE = {"Price agreed": "already in the price", "Not yet in the price": "not yet in the price",
               "Price went the other way": "the price went the other way"}


def _names(xs):
    xs = list(xs)
    return (", ".join(xs[:-1]) + " and " + xs[-1]) if len(xs) > 1 else (xs[0] if xs else "")


def _sectors(D):
    import numpy as np
    by = {}
    for r in D["rows"].values():
        if r["period_ret"] is not None and not np.isnan(r["period_ret"]):
            by.setdefault(r["sector"], []).append(r["period_ret"])
    return sorted(((sec, float(np.mean(v))) for sec, v in by.items()), key=lambda kv: kv[1])


def _movers(D, k=3):
    rows = [r for r in D["rows"].values() if abs(r["period_ret"]) >= 1]
    return sorted(rows, key=lambda r: -abs(r["period_ret"]))[:k]


def summary_draft(D, f):
    """The day's market read, six lines with placeholders, built by rules from
    Tasks 1-6: what happened and where, whether company news explains the big
    moves, the backdrop, what today's news adds, the next three days, and what
    it means. The AI only rewrites it (N11 keeps it to these facts); without AI
    the same lines are printed with the numbers filled in."""
    from .daily_report import (context, evidence_vs_price, ATTENTION_FLAG, FORECAST_WORD, _calm, pick_cards)
    rows = list(D["rows"].values())
    it = D.get("inputs_today") or {}
    secs = _sectors(D)
    weak, strong = secs[:3], secs[-2:][::-1]
    l1 = ("Today the 15 stocks moved {avg_today} on average ({n_up} up, {n_down} down); over {period} they moved "
          "{avg_period}. Weakest: " + _names(f"{sec} ({{sw{i}}})" for i, (sec, _) in enumerate(weak, 1)) + "; "
          + ("strongest" if strong[0][1] > 0 else "held up best") + ": "
          + _names(f"{sec} ({{ss{i}}})" for i, (sec, _) in enumerate(strong, 1)) + ".")
    mv = _movers(D)
    if mv:
        parts = ["Biggest movers over {period}: " + _names(f"{_short(r['name'])} ({{mv{i}}})"
                                                        for i, r in enumerate(mv, 1)) + "."]
        ctx = {r["ticker"]: context(r, D) for r in mv}
        nostory = [r for r in mv if ctx[r["ticker"]]["verdict"] in ("not clearly", "not enough news")]
        wide = [r for r in nostory if ctx[r["ticker"]]["wide"]]
        alone = [r for r in nostory if not ctx[r["ticker"]]["wide"]]
        if wide:
            kinds = {ctx[r["ticker"]]["wide"] for r in wide}
            parts.append(f"{_names(_short(r['name']) for r in wide)} moved without a company news story; the "
                         f"move{'s look' if len(wide) > 1 else ' looks'} {'sector- or market-wide' if len(kinds) > 1 else kinds.pop()}.")
        for word, grp in (("more than", [r for r in alone if not ctx[r["ticker"]]["against"]]),
                          ("against", [r for r in alone if ctx[r["ticker"]]["against"]])):
            if grp:
                parts.append(f"{_names(_short(r['name']) for r in grp)} moved {word} the market without a company "
                             "news story.")
        dis = [r for r in mv if ctx[r["ticker"]]["verdict"] == "no"]
        if dis:
            parts.append(f"For {_names(_short(r['name']) for r in dis)}, the company news and the price disagree.")
        agr = [r for r in mv if ctx[r["ticker"]]["verdict"] == "same direction"]
        if agr:
            parts.append(f"For {_names(_short(r['name']) for r in agr)}, the company news and the price point the "
                         "same way.")
        l2 = " ".join(parts)
    else:
        l2 = "No stock moved more than one percent over {period}, so there is little to explain."
    vix = float(it.get("vix") or 0)
    mood = "calm" if vix < 15 else "normal" if vix < 20 else "nervous" if vix < 25 else "stressed"
    if it.get("options_bearish"):
        opt, join = ("Singapore ETF options lean defensive (put/call {pc_ratio}): more traders are buying protection "
                     "against a fall"), ("but" if mood in ("calm", "normal") else "and")
    elif it.get("options_bullish"):
        opt, join = "Singapore ETF options lean positive (put/call {pc_ratio})", "and"
    else:
        opt, join = "Singapore ETF options are neutral (put/call {pc_ratio})", "and"
    l3 = f"The market backdrop is {mood} (VIX {{vix}}), {join} {opt}."
    sig = sorted([r for r in rows if r["sentiment"] != "neutral"], key=lambda r: -abs(float(r.get("net") or 0)))
    if sig:
        pos = [r for r in sig if r["sentiment"] == "bullish"]
        neg = [r for r in sig if r["sentiment"] == "bearish"]
        def say(xs, cap=4):
            items = [f"{_short(r['name'])} ({PLAIN_PRICE.get(evidence_vs_price(r['sentiment'], r.get('same_day')), 'no same-day price')})"
                     for r in xs[:cap]]
            return _names(items + ([f"{NUM_WORDS[len(xs) - cap]} more"] if len(xs) > cap else []))
        l4 = "Today's news added " + "; ".join(
            ([f"a positive signal for {say(pos)}"] if pos else []) + ([f"a negative signal for {say(neg)}"] if neg else [])) + "."
    else:
        l4 = "Today's news added no positive or negative signal for any stock."
    looks = sorted(D["outlook"].values(), key=lambda o: o["range"]["half_width_pct"])
    k = {c: [r for r in rows if r["t6_class"] == c] for c in ("neutral", "bullish", "bearish")}
    if len(k["neutral"]) == len(rows):
        fc = "The forecast model expects no big move for any of the 15 stocks"
    else:
        who = lambda xs: (_names(_short(r["name"]) for r in xs) if len(xs) <= 3 else f"{NUM_WORDS[len(xs)]} stocks")
        bits = [f"{w} for {who(k[c])}" for c, w in (("bullish", "a rise"), ("bearish", "a fall")) if k[c]]
        fc = "The forecast model leans towards " + "; ".join(bits) + " and no big move for the rest"
        ok = [r for r in k["bullish"] + k["bearish"] if not r["withheld"]]
        fc += (f" (only {who(ok)} with a ✓ record)" if ok else " (none with a ✓ record)")
    l5 = (fc + "; normal 3-day swings run from {swing_low} (" + _short(looks[0]["name"]) + ") to {swing_high} ("
          + _short(looks[-1]["name"]) + "), so a bigger move than a stock's own swing would be unusual.")
    flagged = [r for r in pick_cards(D) if r["attention"] in ATTENTION_FLAG]
    if flagged:
        from .daily_report import flag_reason
        many = len(flagged) > 1
        l6 = ("Flagged ⚑, with a usable model record: " + "; ".join(
            f"{_short(r['name'])}, where {flag_reason(r)}" for r in flagged)
              + (". Their cards are" if many else ". Its card is") + " worth reading first.")
    else:
        l6 = "No stock is flagged ⚑ today."
    if mv:
        quiet = [r for r in mv if context(r, D)["verdict"] in ("not clearly", "not enough news")]
        if quiet:
            l6 += (f" The moves in {_names(_short(r['name']) for r in quiet)} came without company news, so sector "
                   "or market news is the place to check before reacting.")
    calm = _calm(D)
    if calm:
        l6 += f" If you prefer less day-to-day movement, {_names(_short(o['name']) for o in calm)} have the highest chance of no big move."
    return [l1, l2, l3, l4, l5, l6]


def matches_layout(nar, D):
    """A saved narrative fits only the layout it was written for: one summary line
    per market-read line and cards for the stocks the report now shows."""
    from .daily_report import pick_cards
    f, _, _ = build_facts(D)
    summary_ok = not nar.get("summary") or len(nar["summary"]) == len(summary_draft(D, f))
    cards_ok = set(nar.get("cards") or {}) <= {r["ticker"] for r in pick_cards(D)}
    return summary_ok and cards_ok


def revalidate(nar, D):
    """Re-check saved AI text (numbers already filled in) with the content checks and drop any
    block that fails, so a sample never reprints a line a newer check would reject."""
    if not nar or nar.get("mode") != "llm":
        return nar
    from .daily_report import pick_cards, explain_verdict, evidence_vs_price
    f, _, _ = build_facts(D)
    keep = re.compile(r"^(N3 digit|N3 unknown|N4 |N5 )")        # placeholder checks: not applicable after fill

    def content_flags(text, explains=None, price_fact=None):
        return [x for x in check_block(text, f, explains=explains, price_fact=price_fact, max_words=95)
                if not keep.match(x)]

    out = dict(nar, flags=list(nar.get("flags") or []), cards=dict(nar.get("cards") or {}))
    if out.get("summary"):
        fl = [x for s in out["summary"] for x in content_flags(s)]
        if fl:
            out["summary"] = None
            out["flags"] += ["saved summary: " + x for x in fl[:4]]
    for r in pick_cards(D):
        txt = out["cards"].get(r["ticker"])
        if not txt:
            continue
        pf = PLAIN_PRICE.get(evidence_vs_price(r["sentiment"], r.get("same_day")), "no news signal today")
        fl = content_flags(txt, explains=explain_verdict(r), price_fact=pf)
        if fl:
            del out["cards"][r["ticker"]]
            out["flags"] += [f"saved card {r['ticker']}: " + x for x in fl[:3]]
    if out.get("summary"):
        out["summary"] = [plain_terms(s) for s in out["summary"]]
    out["cards"] = {k: plain_terms(v) for k, v in out["cards"].items()}
    if not out["summary"] and not out["cards"]:
        out["mode"] = "template"
    return out


def market_read(D):
    """[(lead, text with numbers)] for the template; same facts as the AI draft."""
    f, _, _ = build_facts(D)
    return [(lead, fill(line, f)) for lead, line in zip(MARKET_READ_LEADS, summary_draft(D, f))]


def _stock_names(D):
    out = set()
    for name in D["names"].values():
        out |= {name.lower(), _short(name).lower()}
    return out


def keeps_to_draft(text, draft, facts, names):
    """N11: the line adds no placeholder or stock name its draft line lacks, and
    keeps the stock names the draft line lists (it may leave out figures: the
    tables show them all)."""
    low, filled = text.lower(), (draft + " " + fill(draft, facts)).lower()
    used = set(_PH.findall(text))
    flags = [f"N11 {{{p}}} is not in its draft line" for p in sorted(used - {"date", "period"})
             if "{" + p + "}" not in draft]
    for nm in sorted(names):
        inside = re.search(rf"\b{re.escape(nm)}\b", low)
        if inside and nm not in filled:
            flags.append(f"N11 adds '{nm}'")
        elif not inside and re.search(rf"\b{re.escape(nm)}\b", draft.lower()):
            flags.append(f"N11 drops '{nm}'")
    return flags


def explain_verdict(r):
    """The 'does the company news explain the move' verdict (daily_report)."""
    from .daily_report import explain_verdict as _ev
    return _ev(r)


def prompt_text(D):
    f, context, cards = build_facts(D)
    lines = [f"{{{k}}} = {v} ({m})" for k, (v, m) in f.items()]
    body = "\n".join(lines) + "\n\nOTHER FACTS (words, may be used as written)\n" + json.dumps(
        {"general": context, "cards": cards}, indent=1, ensure_ascii=False)
    keys = "{" + ", ".join(f'"{c}": "..."' for c in cards) + "}"
    draft = summary_draft(D, f)
    block = "SUMMARY DRAFT (facts checked by code; rewrite these)\n" + "\n".join(
        f"{i}. {x}" for i, x in enumerate(draft, 1))
    texts = {"summary": PROMPT.format(facts=body, draft_block=block,
                                      return_line='Return only JSON: {"summary": [' + ", ".join(['"..."'] * len(draft)) + ']}'),
             "cards": PROMPT.format(facts=body, draft_block="(This answer is the cards only; no summary.)",
                                    return_line='Return only JSON: {"cards": ' + keys + '}')}
    return texts, f, list(cards)


def call_llm(prompt, api_key, settings):
    """One chat completion. gpt-oss models reason before answering, so the
    budget matches Task 5's (4000 tokens), reasoning is kept low and JSON
    mode is requested; a server that rejects those options is retried
    without them."""
    import requests
    import time
    base = {"model": settings["model"], "temperature": settings.get("temperature", 0.2),
            "max_tokens": int(settings.get("max_tokens", {}).get("t5", 4000)),
            "messages": prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]}
    extras = {"response_format": {"type": "json_object"}, "reasoning_effort": "low"}
    err = "no attempt"
    for attempt in range(int(settings.get("max_retries", 3))):
        try:
            resp = requests.post(settings["endpoint"], headers={"Authorization": f"Bearer {api_key}"},
                                 json={**base, **extras}, timeout=settings.get("timeout_s", 60))
            if resp.status_code == 400 and extras:
                extras = {}
                err = f"400 with JSON mode: {resp.text[:120]}"
                continue
            if resp.status_code == 429:
                time.sleep(settings.get("backoff_base_s", 3) * settings.get("backoff_factor", 2) ** attempt)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content") or ""
            if content.strip():
                return content, None
            err = "empty reply"
        except Exception as e:                                   # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return None, err


def _parse(raw):
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


_MOVE_PH = r"\{(?:c\d+_period_move|c\d+_today_move|c\d+_big_day_move|top_period_move|top_today_move|top_period_day_move)\}"


def plain_terms(text):
    """The reader's names for the internal stages ('Task 5' -> the news-and-signals reading, 'Task 6' ->
    the forecast model): a fixed vocabulary map applied by code to every AI text, like the hyphen fix."""
    if not text:
        return text
    t = re.sub(r"\bTask 6 forecasts\b", "the forecast model predicts", text)
    t = re.sub(r"\bTask 6's\b", "the forecast model's", t)
    t = re.sub(r"\bTask 6\b", "the forecast model", t)
    t = re.sub(r"\bTask 5(?:'s)? (reading|readings|rules?)\b", r"the news-and-signals \1", t)
    t = re.sub(r"\bTask 5\b", "the news-and-signals reading", t)
    return re.sub(r"(^|[.!?]\s+)the (forecast model|news-and-signals)", lambda m: m.group(1) + "The " + m.group(2), t)


def tidy(text):
    """'fell over {c1_period_move}' reads as 'more than'; move 'over' after the figure."""
    text = re.sub("[\u2010\u2011\u2012]", "-", str(text or ""))       # non-breaking hyphens

    text = plain_terms(text)

    def repl(m):
        return m.group(1) + ("" if "{period}" in text else " over {period}")
    return re.sub(rf"\bover\s+({_MOVE_PH})", repl, text)


def check_block(text, facts, own=None, max_words=70, explains=None, price_fact=None):
    """Flags for one block of model text (empty list = passes N2-N13)."""
    import task7_generate as g
    from .daily_report import corporate_nouns
    flags, low = [], (text or "").lower()
    if not low.strip():
        return ["N1 empty"]
    for p in g._R1_PATTERNS + g._R2_PATTERNS + g._R3_PATTERNS:
        if re.search(p, corporate_nouns(low)):
            flags.append(f"N2 '{re.search(p, corporate_nouns(low)).group(0)}'")
    names = _PH.findall(text)
    if re.search(r"\d", _ALLOWED.sub("", _PH.sub("", text))):
        flags.append("N3 digit written by the model")
    flags += [f"N3 unknown placeholder {{{n}}}" for n in names if n not in facts]
    if own is not None:
        flags += [f"N4 {{{n}}} belongs to another card" for n in names if re.match(r"c\d+_", n) and not n.startswith(own + "_")]
    if re.search(r"\b(moderate|considerable|significant|strong|high|reliable|trustworthy|meaningful)\s+"
                 r"(weight|confidence|signal|forecast|record)|\b(reliable|trustworthy)\b", low):
        flags.append("N10 gives Task 6 more than little weight")
    for m in _PH.finditer(text):
        val = facts.get(m.group(1), ("",))[0]
        before = low[max(0, m.start() - 40):m.start()]
        near = r"(?:\s+[a-z]+){0,2}\s*\(?$"          # the verb within two words of the figure
        if val.startswith(("-", "−")) and re.search(rf"\b{RISE}\b{near}", before):
            flags.append(f"N5 rise word next to {{{m.group(1)}}} ({val})")
        if val.startswith("+") and re.search(rf"\b{FALL}\b{near}", before):
            flags.append(f"N5 fall word next to {{{m.group(1)}}} ({val})")
    if re.search(r"\b(bullish|bearish)\b", low):
        flags.append("N6 bullish/bearish")
    for sent in re.split(r"(?<=[.!?;])\s+", low):
        if (re.search(r"\b(matching|matches|same as|in line with|mirror(?:s|ing)?|echo(?:es|ing)?)\b", sent)
                and re.search(r"today", sent) and re.search(r"period|five days|5 days|week", sent)):
            flags.append("N8 compares today's move with the period")
            break
    if explains in ("no", "not clearly", "not enough news", "barely moved"):
        for m in re.finditer(r"\b(explain(?:s|ed)?|account(?:s|ed)? for|driven by|drove|due to)\b", low):
            if not re.search(r"\b(not|no|cannot|without|nothing|little|hardly)\b|n't", low[max(0, m.start() - 30):m.start()]):
                flags.append(f"N9 says the news '{m.group(0)}' the move (fact: {explains})")
                break
    if re.search(r"\bper\s?cents?\b|\bpercent\b", low):
        flags.append("N13 a figure written out in words ('percent'); figures must be placeholders")
    if price_fact:
        claims = {"already": re.search(r"\bprice (?:has )?already\b|\balready (?:in the price|reflect|priced|"
                                       r"followed)", low),
                  "not yet": re.search(r"\bnot yet (?:in the price|reflected|followed|shown)\b|"
                                       r"\bprice has(?: not|n't) (?:yet )?(?:reflected|followed|moved)", low),
                  "other way": re.search(r"\b(?:opposite direction|other way|went against the news)\b", low)}
        key = next((k for k in claims if k in price_fact), None)
        for c, m in claims.items():
            if m and c != key:
                flags.append(f"N12 says '{m.group(0)}' but the price fact is '{price_fact}'")
    if len(text.split()) > max_words:
        flags.append(f"N7 {len(text.split())} words")
    return flags


def fill(text, facts):
    """Placeholders -> values; after 'fell' / 'rose' the sign is dropped ('fell 3.1%')."""
    def repl(m):
        val = facts[m.group(1)][0]
        before = text[max(0, m.start() - 40):m.start()].lower()
        if re.search(rf"\b(?:{RISE[3:-1]}|{FALL[3:-1]})\b\s*(?:by\s*)?$", before) and val[:1] in "+-−":
            return val[1:]
        return val
    return _PH.sub(repl, text)


def _read_summary(raw):
    summary = _parse(raw)["summary"]
    summary = [summary] if isinstance(summary, str) else summary
    assert isinstance(summary, list) and summary
    return [tidy(s) for s in summary]


def _read_cards(raw):
    data = _parse(raw)
    cards = data.get("cards", {k: v for k, v in data.items() if re.fullmatch(r"c\d+", str(k))})
    if isinstance(cards, list):
        cards = {f"c{i}": v for i, v in enumerate(cards, 1)}
    assert isinstance(cards, dict)
    return {k: tidy(v) for k, v in cards.items()}


def _feedback(prompt, raw, flags):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your answer broke these checks: " + "; ".join(flags[:12]) +
             ". Fix only those problems and return the full JSON again in the same format."}]


def build(D, api_key=None, settings=None):
    """{'mode', 'model', 'summary': [str] | None, 'cards': {ticker: str}, 'flags': [...], 'retried': [...]}"""
    out = {"mode": "template", "model": (settings or {}).get("model"), "summary": None, "cards": {}, "flags": [],
           "retried": []}
    if not api_key or not settings:
        return out
    prompts, facts, card_keys = prompt_text(D)
    from .daily_report import pick_cards
    rows = pick_cards(D)
    big = max(D["rows"].values(), key=lambda r: abs(r["period_ret"]))
    draft, names = summary_draft(D, facts), _stock_names(D)
    from .daily_report import context as _context
    summary_explains = (None if any(_context(r, D)["verdict"] == "same direction" for r in _movers(D)) else "no")
    attempts = int(settings.get("narrative_attempts", 2))
    raws = []

    def summary_flags(summary):
        fl = [x for s in summary for x in check_block(s, facts, own="summary",
                                                     max_words=70 if len(summary) == len(draft) else 360,
                                                     explains=summary_explains)]
        for i, s in enumerate(summary):
            fl += keeps_to_draft(s, draft[i] if len(summary) == len(draft) else " ".join(draft), facts, names)
        return fl

    def card_flags(cards, k, r):
        from .daily_report import evidence_vs_price
        c = f"c{k}"
        pf = PLAIN_PRICE.get(evidence_vs_price(r["sentiment"], r.get("same_day")), "no news signal today")
        return check_block(cards.get(c, ""), facts, own=c, explains=explain_verdict(r), price_fact=pf)

    # summary: up to two attempts, the second told which checks failed
    msg, final = prompts["summary"], []
    for t in range(attempts):
        raw, err = call_llm(msg, api_key, settings)
        raws.append(f"summary try {t + 1}: {raw if raw is not None else err}")
        if raw is None:
            final = [f"N1 no response ({err})"]
            break
        try:
            summary = _read_summary(raw)
        except Exception as e:                                   # noqa: BLE001
            final = [f"N1 unusable JSON ({type(e).__name__}; reply began: {raw.strip()[:80]!r})"]
            msg = _feedback(prompts["summary"], raw, final)
            continue
        final = summary_flags(summary)
        if not final:
            out["summary"] = [fill(s, facts).strip() for s in summary]
            if t:
                out["retried"].append("summary")
            break
        msg = _feedback(prompts["summary"], raw, final)
    out["flags"] += ["summary: " + x for x in final] if not out["summary"] else []

    # cards: a card that passes is kept; only the failing ones are sent back
    msg, pending = prompts["cards"], {k: ["N1 empty"] for k in range(1, len(rows) + 1)}
    for t in range(attempts):
        raw, err = call_llm(msg, api_key, settings)
        raws.append(f"cards try {t + 1}: {raw if raw is not None else err}")
        if raw is None:
            pending = {k: [f"N1 no response ({err})"] for k in pending}
            break
        try:
            cards = _read_cards(raw)
        except Exception as e:                                   # noqa: BLE001
            pending = {k: [f"N1 unusable JSON ({type(e).__name__})"] for k in pending}
            msg = _feedback(prompts["cards"], raw, ["N1 unusable JSON"])
            continue
        for k in list(pending):
            fl = card_flags(cards, k, rows[k - 1])
            if fl:
                pending[k] = fl
            else:
                out["cards"][rows[k - 1]["ticker"]] = fill(cards[f"c{k}"], facts).strip()
                if t:
                    out["retried"].append(f"c{k}")
                del pending[k]
        if not pending:
            break
        msg = _feedback(prompts["cards"], raw, [f"c{k}: {x}" for k, fl in pending.items() for x in fl])
    out["flags"] += [f"c{k}: {x}" for k, fl in pending.items() for x in fl]
    out["raw"] = "\n\n".join(raws)[:6000]
    if out["summary"] or out["cards"]:
        out["mode"] = "llm"
    return out
