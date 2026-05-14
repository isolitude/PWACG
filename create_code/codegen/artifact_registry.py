#!/usr/bin/env python3
# coding: utf-8
"""Artifact registry — maps artifact names to their generation metadata.

Each artifact knows:
- What template it replaces (legacy jinja2 template path)
- Where the generated file goes
- Which IR fields it needs
- What validation checks apply
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    legacy_template: str
    output_path: str
    ir_fields: tuple[str, ...]
    required_methods: tuple[str, ...]
    description: str


# Artifact registry for S2-S4 migration
ARTIFACTS: dict[str, ArtifactSpec] = {
    # S2: Minimal run templates
    "fit_run": ArtifactSpec(
        name="fit_run",
        legacy_template="templates/fit_run_template.py",
        output_path="run/fit_{generator_id}.py",
        ir_fields=("generator_id", "jinja_fit_info"),
        required_methods=(),
        description="Entry point script that imports and runs the fit object. "
                    "IMPORTANT: Use the exact structure from the fit_run reference pattern in the system prompt. "
                    "The module name comes from context.module (e.g. 'fit'). "
                    "Get CodeScript from ir.jinja_fit_info[module].CodeScript, strip '.py' for import. "
                    "Set ALL run_config and data_config key-values from context on args object. "
                    "The import path must be 'rendered_scripts.<CodeScript without .py>'.",
    ),
    "select_run": ArtifactSpec(
        name="select_run",
        legacy_template="templates/select_run_template.py",
        output_path="run/select_{generator_id}.py",
        ir_fields=("generator_id", "jinja_draw_info"),
        required_methods=(),
        description="Entry point for event selection. "
                    "IMPORTANT: Use the exact structure from the select_run reference pattern in the system prompt. "
                    "CodeScript comes from ir.jinja_draw_info.select.CodeScript[0]. "
                    "Include the Logger class with logconfig_select.json config. "
                    "Copy pwa_info files with os.system at the end.",
    ),
    "draw_wt_run": ArtifactSpec(
        name="draw_wt_run",
        legacy_template="templates/draw_wt_run_template.py",
        output_path="run/draw_wt_{generator_id}.py",
        ir_fields=("generator_id", "jinja_draw_info"),
        required_methods=(),
        description="Entry point for weight drawing. "
                    "IMPORTANT: Use the exact structure from the draw_wt_run reference pattern in the system prompt. "
                    "CodeScript comes from ir.jinja_draw_info.draw_wt.CodeScript[0]. "
                    "Include the Logger class with logconfig_draw.json config. "
                    "Initialize likelihood=0.0 and call draw.get_result_dict() after run_multiprocess.",
    ),
        # S2 Additional run templates
    "batch_run": ArtifactSpec(
        name="batch_run",
        legacy_template="templates/batch_run_template.py",
        output_path="run/batch_{generator_id}.py",
        ir_fields=("generator_id", "jinja_fit_info"),
        required_methods=(),
        description="Entry point for batch scanning. "
                    "IMPORTANT: Use the exact structure from the batch_run reference pattern in the system prompt. "
                    "CodeScript comes from ir.jinja_fit_info.batch.CodeScript. "
                    "Include the Logger class with logconfig_batch.json config.",
    ),
    "lasso_run": ArtifactSpec(
        name="lasso_run",
        legacy_template="templates/lasso_run_template.py",
        output_path="run/lasso_{generator_id}.py",
        ir_fields=("generator_id", "jinja_fit_info"),
        required_methods=(),
        description="Entry point for LASSO regression. "
                    "IMPORTANT: Use the exact structure from the lasso_run reference pattern in the system prompt. "
                    "CodeScript comes from ir.jinja_fit_info.lasso.CodeScript. "
                    "Include the Logger class with logconfig_lasso.json config.",
    ),
    "pull_run": ArtifactSpec(
        name="pull_run",
        legacy_template="templates/pull_run_template.py",
        output_path="run/pull_{generator_id}.py",
        ir_fields=("generator_id", "jinja_fit_info"),
        required_methods=(),
        description="Entry point for pull distribution. "
                    "IMPORTANT: Use the exact structure from the pull_run reference pattern in the system prompt. "
                    "CodeScript comes from ir.jinja_fit_info.pull.CodeScript. "
                    "Include the Logger class with logconfig_fit.json config.",
    ),
    "draw_lh_run": ArtifactSpec(
        name="draw_lh_run",
        legacy_template="templates/draw_lh_run_template.py",
        output_path="run/draw_lh_{generator_id}.py",
        ir_fields=("generator_id", "jinja_draw_info"),
        required_methods=(),
        description="Entry point for likelihood histogram drawing. "
                    "IMPORTANT: Use the exact structure from the draw_lh_run reference pattern in the system prompt. "
                    "CodeScript comes from ir.jinja_draw_info.draw_lh.CodeScript[0]. "
                    "Include the Logger class with logconfig_draw.json config.",
    ),
    "dplot_run": ArtifactSpec(
        name="dplot_run",
        legacy_template="templates/dplot_run_template.py",
        output_path="run/dplot_run_{generator_id}.py",
        ir_fields=("generator_id", "jinja_draw_info"),
        required_methods=(),
        description="Entry point for data plotting with ROOT. "
                    "IMPORTANT: Use the exact structure from the dplot_run reference pattern in the system prompt. "
                    "CodeScript comes from ir.jinja_draw_info.dplot.CodeScript[0]. "
                    "Include the Logger class with logconfig_draw.json config.",
    ),
    # Code templates (S3-S4)
    "select": ArtifactSpec(
        name="select",
        legacy_template="templates/select_template.py",
        output_path="rendered_scripts/select_object_{generator_id}.py",
        ir_fields=("generator_id", "info", "lh_coll", "slit_args_dict",
                   "binding_point", "initial_parameters", "mw_index", "mw_range"),
        required_methods=("thread_likelihood", "compile_func", "run", "run_multiprocess"),
        description="Event selection module. "
                    "MUST generate PWAFunc with weight methods and Control with run_multiprocess. "
                    "CRITICAL: ALL vmap calls in BW helpers MUST use out_axes=1 (NOT np.moveaxis). "
                    "ALL jit() calls MUST include device=self.device.",
    ),
    "draw_wt": ArtifactSpec(
        name="draw_wt",
        legacy_template="templates/draw_wt_template.py",
        output_path="rendered_scripts/draw_wt_object_{generator_id}.py",
        ir_fields=("generator_id", "info", "lh_coll"),
        required_methods=("thread_likelihood", "compile_func", "run", "run_multiprocess"),
        description="Weight distribution drawing. "
                    "MUST generate PWAFunc with weight methods and Control with run_multiprocess. "
                    "CRITICAL: ALL vmap calls in BW helpers MUST use out_axes=1 (NOT np.moveaxis). "
                    "ALL jit() calls MUST include device=self.device.",
    ),
    # S3: Medium templates
    "fit": ArtifactSpec(
        name="fit",
        legacy_template="templates/fit_template.py",
        output_path="rendered_scripts/fit_object_{generator_id}.py",
        ir_fields=("generator_id", "info", "lh_coll", "args_index_collection",
                   "initial_parameters", "mw_index", "mw_range"),
        required_methods=("thread_likelihood", "thread_grad_likelihood",
                          "thread_hvp", "compile_func", "run", "run_multiprocess"),
        description="Main fitting class with likelihood, gradient, and HVP. "
                    "MUST generate the complete PWAFunc class with data_likelihood_<tag>, "
                    "mc_likelihood_<tag>, calculate_<prop>, <prop> propagator, phase, "
                    "jit_request methods for each lh_coll tag. "
                    "MUST generate the complete Control class with thread_likelihood, "
                    "thread_grad_likelihood, thread_hvp, compile_func, run, run_multiprocess. "
                    "Use the EXACT patterns from the system prompt.",
    ),
    "pull": ArtifactSpec(
        name="pull",
        legacy_template="templates/pull_template.py",
        output_path="rendered_scripts/pull_object_{generator_id}.py",
        ir_fields=("generator_id", "info", "lh_coll"),
        required_methods=("thread_likelihood", "compile_func", "run", "run_multiprocess"),
        description="Pull distribution analysis. "
                    "MUST generate PWAFunc and Control classes with pull-specific run() "
                    "that uses Doptimization for pull distribution calculation. "
                    "CRITICAL: ALL calculate_* and lasso_calculate_* methods MUST accept "
                    "Sbc-first argument order: (..., sbc_phi, sbc_f, amp). "
                    "Callsites pass: self.data_phi_kk, self.data_f_kk, self.data_phif0_kk.",
    ),
    "draw_lh": ArtifactSpec(
        name="draw_lh",
        legacy_template="templates/draw_lh_template.py",
        output_path="rendered_scripts/draw_lh_object_{generator_id}.py",
        ir_fields=("generator_id", "info", "lh_coll"),
        required_methods=("thread_likelihood", "compile_func", "run", "run_multiprocess"),
        description="Likelihood histogram drawing. "
                    "MUST generate PWAFunc with weight methods and Control with BIC/mod_weight calculation.",
    ),
    "lasso": ArtifactSpec(
        name="lasso",
        legacy_template="templates/lasso_template.py",
        output_path="rendered_scripts/lasso_object_{generator_id}.py",
        ir_fields=("generator_id", "info", "lh_coll", "lasso_frac_dict"),
        required_methods=("thread_likelihood", "thread_grad_likelihood",
                          "compile_func", "run", "run_multiprocess"),
        description="LASSO regression analysis. "
                    "MUST generate PWAFunc with lasso_calculate methods and Control with LASSO significance calculation. "
                    "CRITICAL: ALL calculate_* and lasso_calculate_* methods MUST accept "
                    "Sbc-first argument order: (..., sbc_phi, sbc_f, amp). "
                    "Callsites pass: self.data_phi_kk, self.data_f_kk, self.data_phif0_kk.",
    ),
    # S4: Large templates
    "batch": ArtifactSpec(
        name="batch",
        legacy_template="templates/batch_template.py",
        output_path="rendered_scripts/batch_object_{generator_id}.py",
        ir_fields=("generator_id", "info", "lh_coll", "args_index_collection",
                   "binding_point", "initial_parameters", "mw_index", "mw_range",
                   "lasso_frac_dict", "sif_free_dict"),
        required_methods=(),
        description="Batch job submission and parameter scanning (948 lines)",
    ),
    "dplot": ArtifactSpec(
        name="dplot",
        legacy_template="templates/dplot_template.py",
        output_path="rendered_scripts/dplot_object_{generator_id}.py",
        ir_fields=("generator_id", "info", "lh_coll", "sbc_collection"),
        required_methods=(),
        description="Data plotting with ROOT histograms (390 lines)",
    ),
    "tensor": ArtifactSpec(
        name="tensor",
        legacy_template="Tensor/RunCacheTensor.py",
        output_path="run/RunCacheTensor.py",
        ir_fields=("generator_id", "CacheTensor"),
        required_methods=(),
        description="Tensor caching runner",
    ),
}


def get_artifact(name: str) -> ArtifactSpec:
    if name not in ARTIFACTS:
        raise KeyError(f"Unknown artifact: {name}. Known: {list(ARTIFACTS.keys())}")
    return ARTIFACTS[name]


def list_artifacts() -> list[str]:
    return list(ARTIFACTS.keys())
