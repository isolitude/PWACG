#!/usr/bin/env python3
"""End-to-end integration tests for the LLM codegen pipeline.

Verifies:
1. Full pipeline: create_all_scripts produces all expected outputs
2. All generated files pass syntax + structural checks
3. Cache store/lookup/golden fallback cycle
4. Minimal likelihood computation on synthetic data
"""
import ast
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check optional dependencies
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    from jax import grad, jit, value_and_grad  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from dlib import dplex  # noqa: F401
    DPLEX_AVAILABLE = True
except ImportError:
    DPLEX_AVAILABLE = False

import numpy as onp

NUMERIC_READY = JAX_AVAILABLE and DPLEX_AVAILABLE


# ---- Pipeline output tests ----

EXPECTED_OUTPUTS = [
    # CodeScripts (rendered_scripts/)
    "rendered_scripts/fit_object_kk.py",
    "rendered_scripts/batch_object_kk.py",
    "rendered_scripts/lasso_object_kk.py",
    "rendered_scripts/pull_object_kk.py",
    "rendered_scripts/select_object_kk.py",
    "rendered_scripts/draw_lh_object_kk.py",
    "rendered_scripts/draw_wt_object_kk.py",
    "rendered_scripts/dplot_object_kk.py",
    # RunScripts (run/)
    "run/fit_kk.py",
    "run/batch_kk.py",
    "run/lasso_kk.py",
    "run/pull_kk.py",
    "run/select_kk.py",
    "run/draw_lh_kk.py",
    "run/draw_wt_kk.py",
    "run/dplot_run_kk.py",
    "run/RunCacheTensor.py",
]


class TestPipelineOutput:
    """Verify create_all_scripts produces all expected files."""

    def test_all_outputs_exist(self):
        missing = [f for f in EXPECTED_OUTPUTS if not Path(f).exists()]
        assert not missing, f"Missing outputs: {missing}"

    @pytest.mark.parametrize("filepath", EXPECTED_OUTPUTS)
    def test_syntax_valid(self, filepath):
        """Each generated file must be valid Python."""
        content = Path(filepath).read_text(encoding="utf-8")
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"{filepath}: syntax error at line {e.lineno}: {e.msg}")

    def test_fit_object_structure(self):
        """fit_object_kk.py must have all required classes and methods."""
        content = Path("rendered_scripts/fit_object_kk.py").read_text(encoding="utf-8")
        tree = ast.parse(content)
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        required = {"PWAFunc", "Control", "ProcessInitializers",
                    "ProcessReturns", "Process_Initializer_Generator", "args"}
        missing = required - classes
        assert not missing, f"fit_object_kk.py missing classes: {missing}"


# ---- Cache system tests ----

class TestCacheSystem:
    """Verify cache store/lookup/golden cycle."""

    def test_cache_store_and_lookup(self):
        from create_code.cache.store import store, lookup

        ir_json = json.dumps({"test": True, "items": [1, 2, 3]}, sort_keys=True)
        key = store(ir_json, "v99", "test-model", "test_artifact", "print('hello')")
        assert len(key) == 64  # SHA256 hex

        result = lookup(ir_json, "v99", "test-model", "test_artifact")
        assert result == "print('hello')"

        # Different prompt version → miss
        result2 = lookup(ir_json, "v98", "test-model", "test_artifact")
        assert result2 is None

    def test_cache_miss(self):
        from create_code.cache.store import lookup

        result = lookup('{"nonexistent": true}', "v1", "unknown", "nonexistent")
        assert result is None

    def test_golden_promote_and_fallback(self):
        from create_code.cache.store import store, promote_to_golden, get_golden

        test_content = "def golden_test():\n    return 42\n"
        ir_json = json.dumps({"golden": "test"}, sort_keys=True)
        key = store(ir_json, "v100", "test-model", "golden_artifact", test_content)
        promote_to_golden("golden_artifact", key)

        try:
            result = get_golden("golden_artifact")
            assert result == test_content
        finally:
            # Cleanup test artifacts
            import shutil
            golden_dir = Path(".llm_cache/golden/golden_artifact")
            by_hash_dir = Path(f".llm_cache/by_hash/{key}")
            if golden_dir.exists():
                shutil.rmtree(golden_dir)
            if by_hash_dir.exists():
                shutil.rmtree(by_hash_dir)

    def test_all_golden_entries_exist(self):
        """Every registered artifact must have a golden fallback."""
        from create_code.cache.store import get_golden
        from create_code.codegen.artifact_registry import list_artifacts

        missing = []
        for name in list_artifacts():
            if get_golden(name) is None:
                missing.append(name)
        assert not missing, f"Missing golden entries: {missing}"


# ---- S6a optimization rule tests ----

S6A_ARTIFACTS = [
    "rendered_scripts/fit_object_kk.py",
    "rendered_scripts/batch_object_kk.py",
    "rendered_scripts/lasso_object_kk.py",
    "rendered_scripts/pull_object_kk.py",
    "rendered_scripts/select_object_kk.py",
    "rendered_scripts/draw_lh_object_kk.py",
    "rendered_scripts/draw_wt_object_kk.py",
    "rendered_scripts/dplot_object_kk.py",
]


class TestS6aRules:
    """Verify JAX optimization rules are applied in all generated files."""

    def _get_ast(self, filepath):
        content = Path(filepath).read_text(encoding="utf-8")
        return ast.parse(content), content

    def _jit_function_bodies(self, filepath):
        """Find functions that get passed to jit() and return their source spans."""
        _, content = self._get_ast(filepath)
        tree = ast.parse(content)
        lines = content.splitlines()

        jit_funcs = set()
        # Find all jit(func, ...) calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "jit":
                if node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Attribute):
                        jit_funcs.add(arg0.attr)
                    elif isinstance(arg0, ast.Name):
                        jit_funcs.add(arg0.id)

        # Find corresponding function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in jit_funcs:
                yield node, lines

    @pytest.mark.parametrize("filepath", S6A_ARTIFACTS)
    def test_rule1_no_onp_in_jit_functions(self, filepath):
        """Rule 1: No onp.eye/zeros/ones inside jit-compiled functions."""
        violations = []
        for func_node, lines in self._jit_function_bodies(filepath):
            start = func_node.lineno
            end = func_node.end_lineno or start
            for i in range(start - 1, end):
                if "onp.eye" in lines[i] or "onp.zeros" in lines[i] or "onp.ones" in lines[i]:
                    violations.append(f"  {func_node.name}:{i + 1}: {lines[i].strip()}")
        assert not violations, f"{filepath} Rule 1 violations:\n" + "\n".join(violations)

    @pytest.mark.parametrize("filepath", S6A_ARTIFACTS)
    def test_rule2_no_moveaxis_after_vmap(self, filepath):
        """Rule 2: No np.moveaxis(vmap(...), 1, 0) pattern."""
        content = Path(filepath).read_text(encoding="utf-8")
        violations = []
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "moveaxis" in stripped and "vmap" in stripped:
                violations.append(f"  {i}: {stripped}")
        assert not violations, f"{filepath} Rule 2 violations:\n" + "\n".join(violations)

    @pytest.mark.parametrize("filepath", S6A_ARTIFACTS)
    def test_rule3_all_jit_have_device(self, filepath):
        """Rule 3: All jit() calls must include device=self.device."""
        content = Path(filepath).read_text(encoding="utf-8")
        violations = []
        for i, line in enumerate(content.splitlines(), 1):
            if "= jit(" in line and "device=" not in line:
                violations.append(f"  {i}: {line.strip()}")
        assert not violations, f"{filepath} Rule 3 violations:\n" + "\n".join(violations)


# ---- Minimal likelihood computation test ----

@pytest.mark.skipif(not NUMERIC_READY, reason="JAX or dplex not available")
class TestSmallDataEndToEnd:
    """Compute likelihood on synthetic data with the generated fit_object_kk."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import importlib
        import rendered_scripts.fit_object_kk as fok
        importlib.reload(fok)
        self.fit_module = fok

        # Synthetic data matching kk channel shape conventions
        rng = onp.random.RandomState(42)
        n_data, n_mc = 50, 200

        class MockCDL:
            def __init__(self):
                self.data_phi_kk = rng.uniform(0.1, 2.0, n_data).astype(onp.float64)
                self.data_f_kk = rng.uniform(0.1, 2.0, n_data).astype(onp.float64)
                self.mc_phi_kk = rng.uniform(0.1, 2.0, n_mc).astype(onp.float64)
                self.mc_f_kk = rng.uniform(0.1, 2.0, n_mc).astype(onp.float64)
                self.data_phif0_kk = rng.uniform(-0.5, 0.5, (2, n_data, 2)).astype(onp.float64)
                self.mc_phif0_kk = rng.uniform(-0.5, 0.5, (2, n_mc, 2)).astype(onp.float64)
                self.data_phif2_kk = rng.uniform(-0.5, 0.5, (5, n_data, 2)).astype(onp.float64)
                self.mc_phif2_kk = rng.uniform(-0.5, 0.5, (5, n_mc, 2)).astype(onp.float64)
                self.wt_data_kk = onp.ones(n_data, dtype=onp.float64)
                self.truth_phi_kk = self.mc_phi_kk.copy()
                self.truth_f_kk = self.mc_f_kk.copy()
                self.truth_phif0_kk = self.mc_phif0_kk.copy()
                self.truth_phif2_kk = self.mc_phif2_kk.copy()

        self.cdl = MockCDL()
        self.test_args = onp.array([
            0.99, 0.05, 1.0, 0.5, 0.3, 1.7, 0.2, 0.4, -0.2, 0.6,
            0.8, 1.27, 0.15, 0.3, -0.1, 0.2, -0.3, 0.4, 0.1, 0.2,
            0.3, 0.4, 0.5, 1.5, 2.1, 2.3, 0.1, 0.15, 0.2, 0.3,
            -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7, 0.8,
            -0.8, 0.9, -0.9, 1.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
            0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8
        ], dtype=onp.float64)

    def test_import_fit_object(self):
        """fit_object_kk module must import successfully."""
        assert hasattr(self.fit_module, "PWAFunc")
        assert hasattr(self.fit_module, "Control")

    def test_create_pwa_func(self):
        """PWAFunc must instantiate successfully."""
        pwa = self.fit_module.PWAFunc(cdl=self.cdl, device_id=None)
        assert hasattr(pwa, "data_likelihood_kk")
        assert hasattr(pwa, "mc_likelihood_kk")

    def test_data_likelihood_finite(self):
        """data_likelihood_kk must return a finite float."""
        pwa = self.fit_module.PWAFunc(cdl=self.cdl, device_id=None)
        lh = pwa.data_likelihood_kk(self.test_args)
        lh_val = float(lh)
        assert onp.isfinite(lh_val), f"data_likelihood is non-finite: {lh_val}"
        assert lh_val != 0.0, f"data_likelihood is zero"

    def test_mc_likelihood_finite(self):
        """mc_likelihood_kk must return a finite float."""
        pwa = self.fit_module.PWAFunc(cdl=self.cdl, device_id=None)
        lh = pwa.mc_likelihood_kk(self.test_args)
        lh_val = float(lh)
        assert onp.isfinite(lh_val), f"mc_likelihood is non-finite: {lh_val}"
        assert lh_val != 0.0, f"mc_likelihood is zero"

    def test_gradient_finite(self):
        """Gradient of data_likelihood must be finite."""
        pwa = self.fit_module.PWAFunc(cdl=self.cdl, device_id=None)
        from jax import grad
        g = grad(pwa.data_likelihood_kk)(self.test_args)
        assert onp.all(onp.isfinite(g)), "gradient contains NaN/inf"

    def test_full_likelihood_computation(self):
        """PWAFunc likelihood computation on synthetic data must be finite."""
        pwa = self.fit_module.PWAFunc(cdl=self.cdl, device_id=None)
        data_lh = float(pwa.data_likelihood_kk(self.test_args))
        mc_lh = float(pwa.mc_likelihood_kk(self.test_args))
        assert onp.isfinite(data_lh), f"data_likelihood is non-finite: {data_lh}"
        assert onp.isfinite(mc_lh), f"mc_likelihood is non-finite: {mc_lh}"
        # Combined likelihood (simplified FOM)
        combined = data_lh + 100.0 * onp.log(mc_lh / 200.0)
        assert onp.isfinite(combined), f"combined likelihood is non-finite: {combined}"
