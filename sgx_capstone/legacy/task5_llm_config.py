# ═══════════════════════════════════════════════════════════════════════════
# T5 LLM LAYER v2 — config-driven  (run AFTER T5 Part B, and after the CONFIG cell)
#
# WHAT THIS REPLACES
#   Three functions that carried hardcoded values inside the Part B cell:
#
#     build_llm_prompt     the whole prompt was an f-string in the code, plus
#                          [:6] [:5] [:5] [:10] [:3] context limits
#     call_groq_llm        model, temperature=0.2, max_tokens=4000,
#                          max_retries=5, backoff 3s x2, timeout=60,
#                          and the endpoint URL
#     validate_llm_output  the ±0.20 confidence tolerance
#     generate_llm_sections (task7_generate)  model, temperature=0.2,
#                          max_tokens=1200, 3 retries, timeout=60
#
#   After this cell, all of those come from config.yaml and the prompt text
#   comes from prompts/task5_prompt.yaml. Nothing else in Part B changes.
#
# WHY A SEPARATE CELL AND NOT AN EDIT INSIDE PART B
#   Part B is ~1,800 lines. Redefining four functions after it loads is a
#   reviewable change; re-pasting the whole cell to alter four functions is not.
#   When the code moves into src/ for the Docker refactor these definitions go
#   into src/task5/llm.py and this cell disappears.
#
# TRACEABILITY
#   Each call returns _meta with the model, the prompt version, the prompt SHA
#   and CONFIG_SHA, so a stored analysis can be tied to the exact wording and
#   the exact configuration that produced it.
# ═══════════════════════════════════════════════════════════════════════════
import json, time, hashlib, os
import requests

import yaml

for _need in ("CFG", "CONFIG_SHA"):
    if _need not in globals():
        raise RuntimeError(f"{_need} is missing — run the CONFIG cell first.")
# Part B ran after the CONFIG cell and re-assigned its own literals, so
# re-publish the config values before redefining the LLM layer.
apply_config(CFG)

_LLM = CFG["llm"]
_LIM = _LLM["t5_limits"]
_T5 = CFG["t5"]


# ── Prompt template, loaded once from YAML ──────────────────────────────────
def load_t5_prompt(path=None, key=None):
    """Return (template, version). Falls back to the in-code copy only if the
    YAML is missing, and says so loudly — a silent fallback would mean the
    stored prompt_version no longer matches the text that was sent."""
    path = path or CFG["paths"]["t5_prompt_yaml"]
    key = key or _LLM["prompt"]["t5_key"]
    with open(path) as f:
        doc = yaml.safe_load(f)
    block = doc[key]
    return block["template"], block.get("version", "yaml")


T5_PROMPT_TEMPLATE, T5_PROMPT_VERSION = load_t5_prompt()
print(f"  Prompt        : {CFG['paths']['t5_prompt_yaml']}  "
      f"version {T5_PROMPT_VERSION}  ({len(T5_PROMPT_TEMPLATE)} chars)")


# ── build_llm_prompt (config-driven) ───────────────────────────────────────
def build_llm_prompt(context, features_result, contradictions, shap_result):
    """Render the Task 5 XAI prompt from the YAML template.

    Substitution is str.replace, not str.format: the JSON schema in the
    template contains literal braces.
    """
    baseline = features_result["baseline_sentiment"]
    confidence = features_result["baseline_confidence"]
    tol = _T5["llm_confidence_tolerance"]
    features = features_result["features"][:_LIM["features"]]

    feature_lines = "\n".join(
        f"  - {f['factor']} ({f['direction']}, weight={f['baseline_weight']}, "
        f"{f.get('share_of_direction', 0):.0%} of {f['direction']} side"
        f"{', DECISIVE' if f.get('flips_call') else ''})"
        for f in features)

    consensus_lines = [
        f"  - {f['label']} (SHAP: {'+' if f['direction'] == 'bullish' else '-'}"
        f"{abs(f['avg_shap']):.3f})"
        for f in shap_result["top_features_consensus"][:_LIM["shap_consensus"]]]

    contra_lines = []
    for c in contradictions["contradictions"][:_LIM["contradictions"]]:
        if c["type"] == "internal":
            contra_lines.append(f"  - [Internal] {c['factor']} (weight: {c['weight']})")
        elif c["type"] == "cross-model":
            contra_lines.append(f"  - [Cross-model] {c['note']}")
        else:
            contra_lines.append(f"  - [Model split] {c['note']}")

    news_titles = (context.get("news") or {}).get("top_titles", [])
    if news_titles:
        news_section = "\n".join(
            f"  {i + 1}. \"{t['title']}\" ({t['sentiment']}, score={t['score']})\n"
            f"     \"{t['content']}\""
            for i, t in enumerate(news_titles[:_LIM["company_news"]]))
    else:
        news_section = "  (no company news today)"

    pol_titles = (context.get("political") or {}).get("top_titles", [])
    if pol_titles:
        pol_section = "\n".join(
            f"  {i + 1}. \"{t['title']}\" ({t['sentiment']}, score={t['score']})"
            for i, t in enumerate(pol_titles[:_LIM["political_news"]]))
    else:
        pol_section = "  (no political news)"

    macro = context.get("macro") or {}
    cr = contradictions.get("consensus_ratio")

    fields = {
        "ticker": context["ticker"], "name": context["name"],
        "sector": context["sector"], "date": context["date"],
        "baseline": baseline, "confidence": f"{confidence}",
        "feature_lines": feature_lines,
        "n_news": str(min(len(news_titles), _LIM["company_news"])),
        "n_political": str(_LIM["political_news"]),
        "news_section": news_section, "pol_section": pol_section,
        "logreg_pred": str(shap_result["predicted_sentiment_logreg"]),
        "logreg_conf": str(shap_result["predicted_confidence_logreg"]),
        "rf_pred": str(shap_result["predicted_sentiment_rf"]),
        "rf_conf": str(shap_result["predicted_confidence_rf"]),
        "consensus_section": "\n".join(consensus_lines) or "  (no consensus)",
        "warning_level": str(contradictions["warning_level"]),
        "consensus_ratio": f"{cr:.0%}" if cr is not None else "n/a",
        "n_internal": str(contradictions["internal_count"]),
        "n_cross_model": str(contradictions["cross_model_count"]),
        "n_model_split": str(contradictions.get("model_split_count", 0)),
        "contra_section": "\n".join(contra_lines) or "  (none)",
        "regime": str(macro.get("regime", "Unknown")),
        "sector_impact": str(macro.get("sector_impact") or "N/A"),
        "conf_tolerance": f"{tol:.2f}",
        "conf_min": f"{max(0.1, confidence - tol):.2f}",
        "conf_max": f"{min(0.99, confidence + tol):.2f}",
    }
    out = T5_PROMPT_TEMPLATE
    for k, v in fields.items():
        out = out.replace("{" + k + "}", v)

    leftover = [k for k in ("{ticker}", "{baseline}", "{feature_lines}")
                if k in out]
    if leftover:
        raise RuntimeError(f"Prompt template still has unfilled placeholders: {leftover}")
    return out


# ── call_groq_llm (config-driven) ──────────────────────────────────────────
def call_groq_llm(prompt, model=None, temperature=None, max_tokens=None,
                  task="t5"):
    """Call the LLM and return parsed JSON (with _meta), or None on failure.

    Every generation parameter comes from config.yaml. Never raises — the
    caller falls back to the rule-based summary.
    """
    model = model or _LLM["model"]
    temperature = _LLM["temperature"] if temperature is None else temperature
    max_tokens = max_tokens or _LLM["max_tokens"][task]
    tries = _LLM["max_retries"]
    base, factor = _LLM["backoff_base_s"], _LLM["backoff_factor"]

    for attempt in range(tries):
        try:
            r = requests.post(
                _LLM["endpoint"],
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature,
                      "max_tokens": max_tokens},
                timeout=_LLM["timeout_s"])

            if r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"]["content"].strip()
                if _LLM.get("strip_code_fences", True):
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0]
                    text = text.strip()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as e:
                    print(f"    ⚠️  JSON parse failed: {e}")
                    print(f"        Response: {text[:200]}")
                    return None
                parsed["_meta"] = {
                    "model": model,
                    "tokens": data.get("usage", {}).get("total_tokens", 0),
                    "temperature": temperature, "max_tokens": max_tokens,
                    "prompt_version": T5_PROMPT_VERSION,
                    "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                    "config_sha": CONFIG_SHA,
                }
                return parsed

            if r.status_code == 429:
                wait = base * (factor ** attempt)
                print(f"    ⏳ Rate limited (429), waiting {wait:.0f}s "
                      f"[{attempt + 1}/{tries}]...")
                time.sleep(wait)
                continue

            print(f"    ❌ {_LLM['provider']} {r.status_code}: {r.text[:200]}")
            return None

        except Exception as e:                                        # noqa: BLE001
            print(f"    ❌ LLM call failed: {type(e).__name__}: {e}")
            return None

    print(f"    ❌ Max retries ({tries}) exceeded")
    return None


# ── validate_llm_output (config-driven tolerance) ──────────────────────────
def validate_llm_output(llm_output, baseline_sentiment, baseline_confidence):
    """Treat the LLM as an untrusted input. Returns (is_valid, notes).

    The confidence tolerance is t5.llm_confidence_tolerance, and the prompt
    quotes the same number — one value, two places it is enforced.
    """
    tol = _T5["llm_confidence_tolerance"]
    notes = []
    if not isinstance(llm_output, dict):
        return False, ["LLM output not a dict"]

    required = ["sentiment", "confidence", "sentiment_score",
                "key_drivers", "risk_factors", "summary"]
    missing = [k for k in required if k not in llm_output]
    if missing:
        return False, [f"Missing key: {k}" for k in missing]

    raw = str(llm_output["sentiment"]).strip()
    if raw.lower() != str(baseline_sentiment).lower().strip():
        return False, [f"Sentiment mismatch: LLM said '{raw}', "
                       f"baseline was '{baseline_sentiment}'"]
    llm_output["sentiment"] = raw.lower()

    try:
        conf = float(llm_output["confidence"])
    except (ValueError, TypeError):
        return False, ["Confidence not a valid float"]
    if abs(conf - baseline_confidence) > tol:
        return False, [f"Confidence out of tolerance (±{tol}): "
                       f"{conf} vs baseline {baseline_confidence}"]
    return True, ["Valid"]



print("Task 5 LLM functions loaded from config and external prompt YAML")
