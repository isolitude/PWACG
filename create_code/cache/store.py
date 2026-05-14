#!/usr/bin/env python3
# coding: utf-8
"""Cache store for LLM-generated artifacts.

Cache key: sha256(ir_json + system_prompt_version + model_id + artifact_name + python_version_minor)

Directory layout:
    .llm_cache/
      by_hash/<hash>/{file_content.py, meta.json, raw_response.json}
      golden/<artifact>/latest -> ../by_hash/<hash>
      prompts/v1/system.md
      prompts/v1/tasks/<artifact>.md
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional


CACHE_ROOT = Path(".llm_cache")
BY_HASH = CACHE_ROOT / "by_hash"
GOLDEN = CACHE_ROOT / "golden"
PROMPTS = CACHE_ROOT / "prompts"


def _cache_key(
    ir_json: str,
    system_prompt_version: str,
    model_id: str,
    artifact_name: str,
) -> str:
    """Deterministic cache key."""
    payload = (
        ir_json
        + "\n"
        + system_prompt_version
        + "\n"
        + model_id
        + "\n"
        + artifact_name
        + "\n"
        + str(sys.version_info.minor)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lookup(
    ir_json: str,
    system_prompt_version: str,
    model_id: str,
    artifact_name: str,
) -> Optional[str]:
    """Check cache for existing artifact. Returns file content or None."""
    key = _cache_key(ir_json, system_prompt_version, model_id, artifact_name)
    cached = BY_HASH / key / "file_content.py"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    return None


def store(
    ir_json: str,
    system_prompt_version: str,
    model_id: str,
    artifact_name: str,
    file_content: str,
    raw_response: Optional[str] = None,
) -> str:
    """Store generated artifact in cache. Returns the cache key."""
    key = _cache_key(ir_json, system_prompt_version, model_id, artifact_name)
    dest = BY_HASH / key
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "file_content.py").write_text(file_content, encoding="utf-8")
    meta = {
        "artifact": artifact_name,
        "model": model_id,
        "prompt_version": system_prompt_version,
        "python_minor": sys.version_info.minor,
    }
    (dest / "meta.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    if raw_response is not None:
        (dest / "raw_response.json").write_text(raw_response, encoding="utf-8")
    return key


def promote_to_golden(artifact_name: str, cache_key: str) -> None:
    """Promote a cached artifact to golden (the fallback reference)."""
    artifact_golden = GOLDEN / artifact_name
    artifact_golden.mkdir(parents=True, exist_ok=True)
    latest = artifact_golden / "latest"
    target = f"../../by_hash/{cache_key}"
    if latest.is_symlink():
        latest.unlink()
    elif latest.is_dir():
        import shutil
        shutil.rmtree(latest)
    elif latest.exists():
        latest.unlink()
    latest.symlink_to(target, target_is_directory=True)


def get_golden(artifact_name: str) -> Optional[str]:
    """Get golden reference artifact. Returns file content or None."""
    golden_dir = GOLDEN / artifact_name / "latest"
    if golden_dir.exists():
        cached = golden_dir / "file_content.py"
        if cached.exists():
            return cached.read_text(encoding="utf-8")
    return None


def get_prompt_version() -> str:
    """Read current prompt version from filesystem.

    Version is determined by the newest vN directory under .llm_cache/prompts/.
    """
    if not PROMPTS.exists():
        return "v0"
    versions = sorted(d.name for d in PROMPTS.iterdir() if d.is_dir() and d.name.startswith("v"))
    return versions[-1] if versions else "v0"
