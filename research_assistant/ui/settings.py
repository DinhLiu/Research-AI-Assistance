from __future__ import annotations

import math
import os
from dataclasses import fields
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import get_args, get_type_hints

from dotenv import dotenv_values, set_key

from research_assistant.config import (
    REPO_ROOT, RetrievalConfig, ExtractionConfig, SynthesisConfig, WritingConfig,
)

STAGES = {"retrieval": RetrievalConfig, "extraction": ExtractionConfig,
          "synthesis": SynthesisConfig, "writing": WritingConfig}
ENV_DEFAULTS = {
    "GROQ_API_KEY": "", "GROQ_MODEL": "llama-3.1-8b-instant",
    "GEMINI_API_KEY": "", "GEMINI_MODEL": "gemini-2.5-flash",
    "OPENAI_API_KEY": "", "OPENAI_MODEL": "gpt-4o-mini",
    "OPENAI_BASE_URL": "https://api.openai.com/v1", "SEMANTIC_SCHOLAR_API_KEY": "",
    "LLM_RPM": "", "LLM_TPM": "", "LLM_RPD": "",
    "LLM_QUOTA_GROUP": "", "LLM_STATE_DIR": "",
}
# Credentials remain in env, never in the public stage config snapshot.
PROVIDERS = ("groq", "openai", "gemini")
PROFILE_FIELDS = ("API_KEY", "MODEL", "LLM_RPM", "LLM_TPM", "LLM_RPD", "LLM_QUOTA_GROUP")
for _stage in STAGES:
    ENV_DEFAULTS[f"RA_{_stage.upper()}_LLM_PROVIDER"] = ""
    for _provider in PROVIDERS:
        for _field in PROFILE_FIELDS + (("BASE_URL",) if _provider == "openai" else ()):
            ENV_DEFAULTS[f"RA_{_stage.upper()}_{_provider.upper()}_{_field}"] = ""

SECRETS = {key for key in ENV_DEFAULTS if key.endswith("API_KEY")}
HIDDEN = {"run_id", "dry_run"}


def schema():
    result = {}
    for stage, cls in STAGES.items():
        defaults = cls()
        hints = get_type_hints(cls)
        result[stage] = []
        for field in fields(cls):
            if field.name in HIDDEN or field.name.endswith("version"):
                continue
            hint = hints[field.name]
            args = get_args(hint)
            nullable = type(None) in args
            base = next((t for t in args if t is not type(None)), hint) if nullable else hint
            kind = {bool: "boolean", int: "integer", float: "number", Path: "path"}.get(base, "text")
            value = getattr(defaults, field.name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                kind, value = "list", ", ".join(value)
            # The UI runs the complete LLM-assisted workflow by default.
            if (stage, field.name) in {("synthesis", "summarize"), ("writing", "use_llm")}:
                value = True
            if stage == "writing" and field.name == "language":
                value = "vi"
            result[stage].append(dict(name=field.name, kind=kind, nullable=nullable, default=value))
    return result


def parse_value(field, value):
    name, kind = field["name"], field["kind"]
    if field["nullable"] and value in (None, ""):
        return None
    if kind == "boolean":
        if value in (True, "true", "True", "1"):
            return True
        if value in (False, "false", "False", "0"):
            return False
        raise ValueError(f"{name}: expected true or false")
    if kind in {"integer", "number"}:
        if isinstance(value, bool):
            raise ValueError(f"{name}: expected a number")
        parsed = int(str(value)) if kind == "integer" else float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"{name}: expected a finite non-negative number")
        zero_ok = name in {"n_query_variants", "max_retries", "max_llm_calls_per_run", "max_logical_calls",
                           "max_repair_calls", "max_http_attempts_per_run", "max_run_tokens", "keyword_weight",
                           "arxiv_delay_s", "llm_min_interval_s"}
        if parsed == 0 and not zero_ok:
            raise ValueError(f"{name}: must be greater than zero")
        return parsed
    if not isinstance(value, str) or len(value) > 4096 or "\n" in value or "\x00" in value:
        raise ValueError(f"{name}: invalid value")
    if kind == "path":
        if not value.strip():
            raise ValueError(f"{name}: a path is required")
        path = Path(value).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path
    if kind == "list":
        return tuple(s.strip() for s in value.split(",") if s.strip())
    return value.strip()


def load_settings(env_path=REPO_ROOT / ".env"):
    saved = dotenv_values(env_path, interpolate=False)
    env = {key: str(saved.get(key, os.environ.get(key, default)) or "") for key, default in ENV_DEFAULTS.items()}
    configs = {}
    for stage, fields_ in schema().items():
        configs[stage] = {f["name"]: saved.get(f"RA_{stage}_{f['name']}".upper(), f["default"]) for f in fields_}
    return env, configs


def validate(env, configs, *, preflight=False):
    if set(env) - ENV_DEFAULTS.keys() or set(configs) - STAGES.keys():
        raise ValueError("Unsupported configuration")
    clean_env = {}
    for key, default in ENV_DEFAULTS.items():
        value = env.get(key, default)
        if not isinstance(value, str) or len(value) > 4096 or any(c in value for c in "\n\r\x00"):
            raise ValueError(f"{key}: invalid value")
        clean_env[key] = value.strip()
    for key in ("LLM_RPM", "LLM_TPM", "LLM_RPD"):
        if clean_env[key] and (not clean_env[key].isdigit() or int(clean_env[key]) <= 0):
            raise ValueError(f"{key}: expected a positive integer")
    for stage in STAGES:
        selected = clean_env[f"RA_{stage.upper()}_LLM_PROVIDER"]
        if selected and selected not in PROVIDERS:
            raise ValueError(f"{stage}: provider must be groq, openai, or gemini")
        for provider in PROVIDERS:
            for name in ("LLM_RPM", "LLM_TPM", "LLM_RPD"):
                key = f"RA_{stage.upper()}_{provider.upper()}_{name}"
                if clean_env[key] and (not clean_env[key].isdigit() or int(clean_env[key]) <= 0):
                    raise ValueError(f"{key}: expected a positive integer")
    parsed = {}
    for stage, fields_ in schema().items():
        values = configs.get(stage, {})
        if not isinstance(values, dict) or set(values) - {f["name"] for f in fields_}:
            raise ValueError(f"{stage}: invalid configuration")
        parsed[stage] = STAGES[stage](**{f["name"]: parse_value(f, values.get(f["name"], f["default"])) for f in fields_})
    r = parsed["retrieval"]
    if r.top_k > r.candidate_pool_size:
        raise ValueError("top_k must not exceed candidate_pool_size")
    if r.year_from and r.year_to and r.year_from > r.year_to:
        raise ValueError("The start year must not exceed the end year")
    if r.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device: choose auto, cpu, cuda, or mps")
    for stage in ("extraction", "synthesis", "writing"):
        if parsed[stage].prefer_provider not in {"gemini", "openai", "groq"}:
            raise ValueError("Provider must be gemini, openai, or groq")
    if preflight:
        if not (r.artifacts_dir / "manifest.json").is_file():
            raise ValueError(f"Missing SPECTER2 data: {r.artifacts_dir / 'manifest.json'}")
        for stage, enabled in {
            "retrieval": r.n_query_variants > 0,
            "extraction": parsed["extraction"].max_llm_calls_per_run > 0,
            "synthesis": parsed["synthesis"].summarize and not parsed["synthesis"].strict_evidence,
            "writing": parsed["writing"].use_llm,
        }.items():
            if not enabled:
                continue
            effective = stage_environment(stage, clean_env)
            selected = clean_env[f"RA_{stage.upper()}_LLM_PROVIDER"]
            has_key = any(effective.get(f"{p.upper()}_API_KEY") for p in PROVIDERS)
            if selected and not has_key:
                raise ValueError(f"{stage}: enter an API key for {selected}")
            if not has_key and stage in {"synthesis", "writing"}:
                raise ValueError("Enter at least one API key to use an LLM, or disable summarize and use_llm.")
            if has_key and not all(effective.get(k) for k in ("LLM_RPM", "LLM_TPM", "LLM_RPD")):
                raise ValueError(f"{stage}: enter LLM_RPM, LLM_TPM, and LLM_RPD according to your quota.")
    return clean_env, parsed


def save_settings(env, configs, env_path=REPO_ROOT / ".env"):
    clean_env, parsed = validate(env, configs)
    # Preserve unrelated keys/comments; replace atomically with owner-only permissions.
    with NamedTemporaryFile(mode="w", dir=env_path.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(env_path.read_text(encoding="utf-8") if env_path.exists() else "")
        temp = Path(tmp.name)
    try:
        values = dict(clean_env)
        for stage, fields_ in schema().items():
            for f in fields_:
                value = getattr(parsed[stage], f["name"])
                if isinstance(value, tuple):
                    value = ",".join(value)
                values[f"RA_{stage}_{f['name']}".upper()] = "" if value is None else str(value)
        for key, value in values.items():
            set_key(temp, key, value, quote_mode="always")
        os.chmod(temp, 0o600)
        os.replace(temp, env_path)
    finally:
        temp.unlink(missing_ok=True)


def stage_environment(stage, env):
    """Resolve a UI stage profile. Blank fields inherit shared provider settings."""
    result = dict(env)
    provider = env.get(f"RA_{stage.upper()}_LLM_PROVIDER", "")
    if not provider:
        return result  # Legacy settings retain their provider cascade.
    prefix = f"RA_{stage.upper()}_{provider.upper()}_"
    for other in PROVIDERS:
        if other != provider:
            result[f"{other.upper()}_API_KEY"] = ""
    result["GOOGLE_API_KEY"] = ""
    for field in PROFILE_FIELDS + (("BASE_URL",) if provider == "openai" else ()):
        target = field if field.startswith("LLM_") else f"{provider.upper()}_{field}"
        if env.get(prefix + field):
            result[target] = env[prefix + field]
    return result
