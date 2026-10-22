"""Validated runtime settings. Secrets are read only from the environment."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(os.environ.get("SGX_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def load_config():
    path = ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("t5", {}).get("model_version") != "v6":
        raise ValueError("Expected the reviewed Task 5 v6 configuration")
    if cfg.get("t6", {}).get("expected_forward_horizon") != 3:
        raise ValueError("Expected a three trading session Task 6 target")
    raw = path.read_text().replace("${project_root}", str(ROOT)).replace(
        "{t5_model_version}", cfg["t5"]["model_version"])
    cfg = yaml.safe_load(raw)
    cfg["paths"]["project_root"] = str(ROOT)
    for name in ("t5_artifacts", "t6_artifacts", "t5_prompt_yaml", "t7_prompt_yaml"):
        if not Path(cfg["paths"][name]).exists():
            raise FileNotFoundError(f"Required {name}: {cfg['paths'][name]}")
    inputs = [path, *sorted((ROOT / "settings").glob("*.json")),
              Path(cfg["paths"]["t5_prompt_yaml"]), Path(cfg["paths"]["t7_prompt_yaml"]),
              Path(cfg["paths"]["t5_artifacts"]), Path(cfg["paths"]["t6_artifacts"])]
    cfg["input_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in inputs}
    sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return cfg, sha


def snapshot_config(sha):
    """Keep the exact non-secret source YAML used for a live output."""
    folder = ROOT / "outputs" / "config_snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"config_{sha}.yaml"
    source = (ROOT / "config.yaml").read_bytes()
    if target.exists() and target.read_bytes() != source:
        raise RuntimeError("Configuration hash collision or changed snapshot")
    if not target.exists():
        target.write_bytes(source)
    cfg, current_sha = load_config()
    if sha != current_sha:
        raise RuntimeError("Inputs changed after configuration was loaded")
    (folder / f"manifest_{sha}.json").write_text(json.dumps(cfg, indent=2))
    for relative in cfg["input_sha256"]:
        if relative.startswith(("settings/", "prompts/")):
            copy = folder / sha / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes((ROOT / relative).read_bytes())
    return target


def apply_config(cfg, target):
    """Legacy compatibility for selected notebook functions; called after loading them."""
    scalar = {
        "T5_MODEL_VERSION": ("t5", "model_version"),
        "RULE_LAYER_VERSION": ("t5", "rule_layer_version"),
        "MAX_CONFIDENCE": ("t5", "max_confidence"),
        "NET_SCORE_THRESHOLD": ("t5", "net_score_threshold"),
        "MARKET_WIDE_SCALE": ("t5", "market_wide_scale"),
        "EVENT_SCALE": ("t5", "event_scale"),
        "EVENT_TRAIL_DAYS": ("t5", "event_trail_days"),
        "EVENT_EXCESS_MIN": ("t5", "event_excess_min"),
        "EVENT_MIN_HISTORY": ("t5", "event_min_history"),
        "LLM_CONFIDENCE_TOLERANCE": ("t5", "llm_confidence_tolerance"),
        "CHAMPION_MARGIN": ("t6", "champion_margin"),
        "CALIB_SPLITS": ("t6", "calib_splits"),
        "PKL_FALLBACK_CHAMPION": ("t6", "fallback_champion"),
        "SUPPRESS_EDGE": ("t7", "suppress_edge"),
        "TRACK_RECORD_WINDOW": ("t7", "track_record_window_days"),
        "ATTENTION_MIN_T6_PROB": ("t7", "attention_min_t6_prob"),
        "GROQ_MODEL": ("llm", "model"),
    }
    for name, (section, key) in scalar.items():
        target[name] = cfg[section][key]
    for name, section, key in [
        ("MARKET_WIDE_FEATURES", "t5", "market_wide_features"),
        ("BULLISH_EVENTS", "t5", "bullish_events"),
        ("BEARISH_EVENTS", "t5", "bearish_events"),
    ]:
        target[name] = set(cfg[section][key])
    target["ARTIFACTS_PATH"] = cfg["paths"]["t5_artifacts"]
    target["ARTIFACTS_PATH_T6"] = cfg["paths"]["t6_artifacts"]
    target["FEATURE_CACHE"] = cfg["paths"]["feature_cache"]
    target["RESEARCH_DIR"] = cfg["paths"]["research_dir"]
    target["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
    return target


def configure_report_modules(cfg):
    import pickle
    import task7_report as report
    import task7_generate as generate
    # These are learned label boundaries, loaded from the actual serving artifacts.
    # The pickles are bundled, trusted project files; never load a user supplied pickle.
    with open(cfg["paths"]["t5_artifacts"], "rb") as stream:
        t5 = pickle.load(stream)
    with open(cfg["paths"]["t6_artifacts"], "rb") as stream:
        t6 = pickle.load(stream)
    report.MAJORITY_BASELINE = None
    report.SUPPRESS_EDGE = cfg["t7"]["suppress_edge"]
    report.ATTENTION_MIN_T6_ACC = 1.0
    report.ATTENTION_MIN_T6_PROB = cfg["t7"]["attention_min_t6_prob"]
    report.TRACK_RECORD_WINDOW = cfg["t7"]["track_record_window_days"]
    report.PRICE_HISTORY_START = cfg["t7"]["price_history_start"]
    report.T5_THRESHOLD_DEFAULT = float(t5["return_threshold"])
    report.T6_THRESHOLD_DEFAULT = float(t6["return_threshold"])
    report.ATTENTION_ARCHETYPES = set(cfg["t7"]["attention_archetypes"])
    report.LOW_INFORMATION_ARCHETYPES = set(cfg["t7"]["low_information_archetypes"])
    generate.SUPPRESS_EDGE = cfg["t7"]["suppress_edge"]
    generate.DISCLAIMER = cfg["t7"]["disclaimer"]
    generate.DEFAULT_YAML_PATH = cfg["paths"]["t7_prompt_yaml"]
    generate.LLM_SETTINGS = cfg["llm"]
    generate.MAJORITY_BASELINE = report.MAJORITY_BASELINE
    generate.T6_THRESHOLD_DEFAULT = report.T6_THRESHOLD_DEFAULT
    generate.LOW_INFORMATION_ARCHETYPES = report.LOW_INFORMATION_ARCHETYPES


def audit_serving_artifact(cfg):
    """Inspect trusted bundled T6 artifact; this does not establish live skill."""
    import pickle
    with open(cfg["paths"]["t6_artifacts"], "rb") as stream:
        art = pickle.load(stream)
    estimators = ("logreg_calibrated", "rf_calibrated", "xgb_calibrated")
    prefit = [key for key in estimators if art.get(key) is not None
              and str(getattr(art[key], "cv", None)) == "prefit"]
    return {"model_version": str(art.get("model_version", "unknown")),
            "serving_sha256": hashlib.sha256(Path(cfg["paths"]["t6_artifacts"]).read_bytes()).hexdigest(),
            "source_sha256": art.get("source_artifact_sha256", "not recorded"),
            "trained_at": str(art.get("trained_at", "unknown")),
            "calibration": str(art.get("calibration", "missing")),
            "prefit_estimators": prefit,
            "passes_metadata_preflight": (
                str(art.get("calibration", "")).startswith("sigmoid, out-of-fold")
                and not prefit)}
