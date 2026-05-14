#!/usr/bin/env python3
# coding: utf-8
"""Code generator: orchestrates IR → cache → LLM → validate → write.

S2-S4 pipeline. Each artifact goes through:
    1. Cache lookup (0 token if hit)
    2. LLM call with system prompt + task prompt + IR JSON
    3. Syntax → types → structural validation
    4. On failure: retry with error feedback (max 3 rounds)
    5. On persistent failure: fallback to golden
    6. On success: write to output path, promote to golden
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .artifact_registry import get_artifact, ArtifactSpec
from ..cache.store import lookup, store, promote_to_golden, get_golden, get_prompt_version
from ..validate import check_syntax, check_types, check_structural

log = logging.getLogger(__name__)


class CodeGenerator:
    def __init__(
        self,
        *,
        llm: Optional[LLMClient] = None,
        system_prompt_path: Path = Path("create_code/codegen/prompts/system.md"),
        prompt_version: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.llm = llm
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")
        self.prompt_version = prompt_version or get_prompt_version()
        self.dry_run = dry_run

    def generate(self, ir, artifact_name: str, *, extra_context: Optional[dict] = None) -> str:
        """Generate one artifact from IR.

        Returns the file content (from cache, LLM, or golden fallback).
        """
        spec = get_artifact(artifact_name)
        ir_json = json.dumps(ir.model_dump(by_alias=True, mode="json"), sort_keys=True)

        # 1. Cache lookup
        cached = lookup(ir_json, self.prompt_version, self.llm.model if self.llm else "offline", artifact_name)
        if cached is not None:
            log.info(f"[{artifact_name}] cache hit — 0 token")
            return cached

        # 2. Try LLM generation
        if self.llm is not None and not self.dry_run:
            for attempt in range(3):
                try:
                    file_content = self._call_llm(ir, ir_json, spec, extra_context=extra_context)
                    # 3. Validate
                    if self._validate(file_content, spec):
                        # 4. Cache & promote
                        cache_key = store(
                            ir_json, self.prompt_version,
                            self.llm.model, artifact_name, file_content,
                        )
                        promote_to_golden(artifact_name, cache_key)
                        return file_content
                except Exception as e:
                    log.warning(f"[{artifact_name}] attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        break

        # 5. Fallback to golden
        golden = get_golden(artifact_name)
        if golden is not None:
            log.warning(f"[{artifact_name}] falling back to golden")
            return golden

        raise RuntimeError(
            f"[{artifact_name}] generation failed — no LLM, no golden fallback. "
            f"Set DEEPSEEK_API_KEY or ensure golden exists."
        )

    def _call_llm(self, ir, ir_json: str, spec: ArtifactSpec, *, extra_context: Optional[dict] = None) -> str:
        """Call LLM with system prompt + task + IR."""
        task_prompt = self._build_task_prompt(ir, spec, extra_context=extra_context)
        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=task_prompt,
        )
        return response["file_content"]

    def _build_task_prompt(self, ir, spec: ArtifactSpec, *, extra_context: Optional[dict] = None) -> str:
        """Build the user prompt for LLM."""
        # Extract only the IR fields this artifact needs
        ir_subset: dict = {}
        ir_full = ir.model_dump(by_alias=True, mode="json")
        for field in spec.ir_fields:
            if field in ir_full:
                ir_subset[field] = ir_full[field]

        task: dict = {
            "artifact": spec.name,
            "description": spec.description,
            "output_path": spec.output_path.format(generator_id=ir.generator_id),
            "required_methods": list(spec.required_methods),
            "ir": ir_subset,
            "instructions": (
                f"Generate the complete Python file for '{spec.name}'. "
                f"This replaces the jinja2 template at '{spec.legacy_template}'. "
                f"Output the file content in the 'file_content' field of your JSON response. "
                f"The file must have ALL required methods: {list(spec.required_methods)}. "
                f"Apply ALL S6a JAX optimization rules from the system prompt."
            ),
        }
        if extra_context:
            task["context"] = extra_context

        return json.dumps(task, ensure_ascii=False, sort_keys=True)

    def _validate(self, file_content: str, spec: ArtifactSpec) -> bool:
        """Run fail-fast validation pipeline."""
        # 1. Syntax
        syn = check_syntax(file_content)
        if not syn.passed:
            log.warning(f"[{spec.name}] syntax error line {syn.error_line}: {syn.error}")
            return False

        # 2. Types (lightweight import check)
        typ = check_types(file_content)
        if not typ.passed:
            log.warning(f"[{spec.name}] type errors: {typ.errors}")
            return False

        # 3. Structural
        struct = check_structural(file_content, required_methods=spec.required_methods)
        if not struct.passed:
            log.warning(f"[{spec.name}] structural errors: {struct.errors}")
            return False

        return True
