#!/usr/bin/env python3
"""Numeric equivalence testing for LLM-generated code.

S3 validation: verify LLM-generated fit/pull/draw_lh/lasso artifacts
produce numerically equivalent results vs jinja2 reference.

Strategy:
1. Structural: verify generated files have expected classes/methods
2. Import: verify generated files can be imported
3. Numeric: compare thread_likelihood / thread_grad_likelihood output
   vs golden reference (requires JAX + dplex + GPU or CPU backend)
"""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if JAX is available
try:
    import jax  # noqa: F401
    import jax.numpy as np  # noqa: F401
    from jax import grad  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Check if dplex is available
try:
    from dlib import dplex  # noqa: F401
    DPLEX_AVAILABLE = True
except ImportError:
    DPLEX_AVAILABLE = False

import numpy as onp  # noqa: E402

NUMERIC_READY = JAX_AVAILABLE and DPLEX_AVAILABLE


# ---- Structural tests (always run) ----

S3_ARTIFACTS = ["fit", "pull", "draw_lh", "lasso"]
S4_ARTIFACTS = ["batch", "dplot"]

REQUIRED_CLASSES = {
    "fit": {"Control", "PWAFunc", "ProcessInitializers", "ProcessReturns",
            "Process_Initializer_Generator", "args"},
    "pull": {"Control", "PWAFunc", "ProcessInitializers", "ProcessReturns",
             "Process_Initializer_Generator", "args"},
    "draw_lh": {"Control", "PWAFunc", "ProcessInitializers", "ProcessReturns",
                "Process_Initializer_Generator", "args"},
    "lasso": {"Control", "PWAFunc", "ProcessInitializers", "ProcessReturns",
              "Process_Initializer_Generator", "args"},
    "batch": {"Control", "ProcessInitializers", "ProcessReturns",
              "Process_Initializer_Generator", "args"},
    "dplot": {"Control", "ProcessInitializers", "ProcessReturns",
              "Process_Initializer_Generator", "args"},
}

REQUIRED_METHODS = {
    "fit": {"thread_likelihood", "thread_grad_likelihood", "thread_hvp",
            "compile_func", "run", "run_multiprocess"},
    "pull": {"thread_likelihood", "compile_func", "run", "run_multiprocess"},
    "draw_lh": {"thread_likelihood", "compile_func", "run", "run_multiprocess"},
    "lasso": {"thread_likelihood", "thread_grad_likelihood",
              "compile_func", "run", "run_multiprocess"},
    "batch": {"thread_likelihood", "compile_func", "run", "run_multiprocess"},
    "dplot": {"thread_likelihood", "compile_func", "run", "run_multiprocess"},
}


class TestGeneratedStructure:
    """Verify all generated artifacts have correct structure."""

    @pytest.mark.parametrize("artifact", S3_ARTIFACTS + S4_ARTIFACTS)
    def test_syntax_valid(self, artifact):
        """Generated Python files must be syntactically valid."""
        path = f"rendered_scripts/{artifact}_object_kk.py"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run create_all_scripts.py first")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"{artifact}: syntax error: {e}")

    @pytest.mark.parametrize("artifact", S3_ARTIFACTS + S4_ARTIFACTS)
    def test_required_classes(self, artifact):
        """Generated files must contain all expected classes."""
        path = f"rendered_scripts/{artifact}_object_kk.py"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run create_all_scripts.py first")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        missing = REQUIRED_CLASSES[artifact] - classes
        assert not missing, f"{artifact}: missing classes: {missing}"

    @pytest.mark.parametrize("artifact", S3_ARTIFACTS)
    def test_required_methods(self, artifact):
        """Generated S3 files must contain all required methods."""
        path = f"rendered_scripts/{artifact}_object_kk.py"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run create_all_scripts.py first")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        missing = REQUIRED_METHODS[artifact] - funcs
        assert not missing, f"{artifact}: missing methods: {missing}"

    @pytest.mark.parametrize("artifact", S4_ARTIFACTS)
    def test_required_methods_s4(self, artifact):
        """Generated S4 files: check at minimum run/run_multiprocess exist."""
        path = f"rendered_scripts/{artifact}_object_kk.py"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run create_all_scripts.py first")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        # S4 only requires run and run_multiprocess for now (full spec pending)
        for m in ("run", "run_multiprocess"):
            assert m in funcs, f"{artifact}: missing method: {m}"

    @pytest.mark.parametrize("artifact", ["fit"])
    def test_tag_specific_likelihood_methods(self, artifact):
        """Fit artifact must have data_likelihood_<tag> and mc_likelihood_<tag>."""
        path = f"rendered_scripts/{artifact}_object_kk.py"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run create_all_scripts.py first")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

        # Find all tag-specific likelihood methods
        data_lh = [f for f in funcs if f.startswith("data_likelihood_")]
        mc_lh = [f for f in funcs if f.startswith("mc_likelihood_")]
        assert data_lh, "No data_likelihood_<tag> methods found"
        assert mc_lh, "No mc_likelihood_<tag> methods found"
        # Tags should match
        data_tags = {f.split("data_likelihood_", 1)[1] for f in data_lh}
        mc_tags = {f.split("mc_likelihood_", 1)[1] for f in mc_lh}
        assert data_tags == mc_tags, f"Tag mismatch: data={data_tags}, mc={mc_tags}"

    @pytest.mark.parametrize("artifact", ["fit"])
    def test_propagator_methods(self, artifact):
        """Fit artifact must have propagator functions."""
        path = f"rendered_scripts/{artifact}_object_kk.py"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run create_all_scripts.py first")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

        # At least one calculate_ method
        calc_methods = [f for f in funcs if f.startswith("calculate_")]
        assert calc_methods, "No calculate_<prop> methods found"

        # At least one propagator method (BW, flatte, etc.)
        prop_methods = [f for f in funcs if any(
            f.startswith(p) for p in ("BW", "flatte")
        ) and not f.startswith("calculate_")]
        assert prop_methods, f"No propagator methods found in: {sorted(funcs)}"

    @pytest.mark.parametrize("artifact", ["fit"])
    def test_s6a_rule1_no_onp_in_jit(self, artifact):
        """S6a Rule 1: No onp.eye() or onp.zeros() inside @jit functions."""
        path = f"rendered_scripts/{artifact}_object_kk.py"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run create_all_scripts.py first")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # This is a heuristic check — verify the pattern isn't used
        # A more thorough check would require parsing decorators
        if "onp.eye()" in content or "onp.zeros()" in content:
            pytest.fail("S6a Rule 1 violation: onp.eye()/onp.zeros() found in generated code")

    @pytest.mark.parametrize("artifact", ["fit"])
    def test_s6a_rule2_no_moveaxis_vmap(self, artifact):
        """S6a Rule 2: No np.moveaxis(vmap(...), 1, 0) pattern."""
        path = f"rendered_scripts/{artifact}_object_kk.py"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run create_all_scripts.py first")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if "moveaxis(vmap(" in content:
            pytest.fail("S6a Rule 2 violation: moveaxis(vmap(...)) pattern found")


# ---- Import tests (require JAX) ----

class TestGeneratedImport:
    """Verify generated files can be imported (requires JAX)."""

    @pytest.mark.skipif(not NUMERIC_READY, reason="JAX and dplex required")
    @pytest.mark.parametrize("artifact", S3_ARTIFACTS)
    def test_import_artifact(self, artifact):
        """Generated artifacts should be importable."""
        import importlib
        mod_name = f"rendered_scripts.{artifact}_object_kk"
        try:
            mod = importlib.import_module(mod_name)
            assert mod is not None
        except Exception as e:
            pytest.fail(f"Failed to import {artifact}: {e}")


# ---- Numeric equivalence tests (require JAX + dplex) ----

@pytest.fixture(scope="class")
def mock_cdl():
    """Create synthetic mock CDL data with correct tensor shapes.

    Key shape conventions (verified against jinja2 reference):
    - Amp arrays: (damp, n_events, 2) — damp FIRST, then events, then 2 components
    - Sbc arrays (phi_kk, f_kk): (n_events,) — 1D per event
    - weight arrays: (n_events,) — 1D per event
    """
    n_data, n_mc = 100, 1000
    rng = onp.random.RandomState(42)

    class MockCDL:
        def __init__(self):
            self.data_phi_kk = rng.uniform(0.1, 2.0, n_data).astype(onp.float64)
            self.data_f_kk = rng.uniform(0.1, 2.0, n_data).astype(onp.float64)
            self.mc_phi_kk = rng.uniform(0.1, 2.0, n_mc).astype(onp.float64)
            self.mc_f_kk = rng.uniform(0.1, 2.0, n_mc).astype(onp.float64)
            # phif0_kk: damp=2 (from IR: BW_flatte980 and BW_BW on phif0)
            self.data_phif0_kk = rng.uniform(-0.5, 0.5, (2, n_data, 2)).astype(onp.float64)
            self.mc_phif0_kk = rng.uniform(-0.5, 0.5, (2, n_mc, 2)).astype(onp.float64)
            # phif2_kk: damp=5 (from IR: BW_flatte1270 and BW_BW on phif2)
            self.data_phif2_kk = rng.uniform(-0.5, 0.5, (5, n_data, 2)).astype(onp.float64)
            self.mc_phif2_kk = rng.uniform(-0.5, 0.5, (5, n_mc, 2)).astype(onp.float64)
            self.wt_data_kk = onp.ones(n_data, dtype=onp.float64)
            # Truth data for lasso calculations
            self.truth_phi_kk = self.mc_phi_kk.copy()
            self.truth_f_kk = self.mc_f_kk.copy()
            self.truth_phif0_kk = self.mc_phif0_kk.copy()
            self.truth_phif2_kk = self.mc_phif2_kk.copy()
    return MockCDL()


@pytest.fixture(scope="class")
def test_args():
    """59 test parameter values matching the kk fit arg layout."""
    return onp.array([
        0.99, 0.05, 1.0, 0.5, 0.3, 1.7, 0.2, 0.4, -0.2, 0.6,
        0.8, 1.27, 0.15, 0.3, -0.1, 0.2, -0.3, 0.4, 0.1, 0.2,
        0.3, 0.4, 0.5, 1.5, 2.1, 2.3, 0.1, 0.15, 0.2, 0.3,
        -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7, 0.8,
        -0.8, 0.9, -0.9, 1.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
        0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8
    ], dtype=onp.float64)


class TestNumericEquivalence:
    """Compare LLM-generated code output with jinja2 golden reference."""


    @pytest.fixture(scope="class")
    def pwa_funcs(self, mock_cdl):
        """Import both LLM and jinja2 PWAFunc classes and instantiate with mock data."""
        if not NUMERIC_READY:
            pytest.skip("JAX and dplex required for numeric tests")

        import importlib.util

        llm_path = "rendered_scripts/fit_object_kk.py"
        j2_path = "/tmp/fit_jinja2_ref.py"

        if not os.path.exists(llm_path):
            pytest.skip(f"{llm_path} not found — run create_all_scripts.py first")
        if not os.path.exists(j2_path):
            pytest.skip(f"{j2_path} not found — generate jinja2 reference first")

        spec_llm = importlib.util.spec_from_file_location("fit_llm_numtest", llm_path)
        mod_llm = importlib.util.module_from_spec(spec_llm)
        sys.modules["fit_llm_numtest"] = mod_llm
        spec_llm.loader.exec_module(mod_llm)

        spec_j2 = importlib.util.spec_from_file_location("fit_j2_numtest", j2_path)
        mod_j2 = importlib.util.module_from_spec(spec_j2)
        sys.modules["fit_j2_numtest"] = mod_j2
        spec_j2.loader.exec_module(mod_j2)

        pwa_llm = mod_llm.PWAFunc(cdl=mock_cdl, device_id=None)
        pwa_j2 = mod_j2.PWAFunc(cdl=mock_cdl, device_id=None)

        return pwa_llm, pwa_j2

    def _make_params(self, args):
        """Build parameter dicts for amplitude computation. Returns (phi_m, phi_w, params_dict)."""
        phi_m = np.array([1.02])
        phi_w = np.array([0.004])
        p = {}
        p['f980_m'] = np.array([args[0]]); p['f980_g'] = np.array([args[1]]); p['f980_r'] = np.array([args[2]])
        p['f980_c'] = np.array([0.1, args[3]]).reshape(-1, 2)
        p['f980_t'] = np.array([0.1, args[4]]).reshape(-1, 2)
        p['f0_m'] = np.array([args[5]]); p['f0_w'] = np.array([args[6]])
        p['f0_c'] = np.array([args[7], args[8]]).reshape(-1, 2)
        p['f0_t'] = np.array([args[9], args[10]]).reshape(-1, 2)
        p['f1270_m'] = np.array([args[11]]); p['f1270_w'] = np.array([args[12]])
        p['f1270_c'] = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5)
        p['f1270_t'] = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5)
        p['f2_m'] = np.array([args[23], args[24], args[25]])
        p['f2_w'] = np.array([args[26], args[27], args[28]])
        p['f2_c'] = np.array([args[29], args[30], args[31], args[32], args[33], args[34],
                              args[35], args[36], args[37], args[38], args[39], args[40],
                              args[41], args[42], args[43]]).reshape(-1, 5)
        p['f2_t'] = np.array([args[44], args[45], args[46], args[47], args[48], args[49],
                              args[50], args[51], args[52], args[53], args[54], args[55],
                              args[56], args[57], args[58]]).reshape(-1, 5)
        return phi_m, phi_w, p

    def _compute_combined(self, pwa, args, is_llm, mc=False):
        """Compute combined amplitude for a PWAFunc instance.

        Note: LLM and jinja2 versions have different argument orders in calculate_*:
        - LLM: calculate_*(..., amp, Sbc_phi, Sbc_f)
        - jinja2: calculate_*(..., Sbc_phi, Sbc_f, amp)
        """
        phi_m, phi_w, p = self._make_params(args)
        pref = "mc_" if mc else "data_"

        if is_llm:
            d1 = pwa.calculate_BW_flatte980(phi_m, phi_w, p['f980_m'], p['f980_g'], p['f980_r'],
                    p['f980_c'], p['f980_t'],
                    getattr(pwa, pref+'phif0_kk'), getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'))
            d2 = pwa.calculate_BW_BW(phi_m, phi_w, p['f0_m'], p['f0_w'],
                    p['f0_c'], p['f0_t'],
                    getattr(pwa, pref+'phif0_kk'), getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'))
            d3 = pwa.calculate_BW_flatte1270(phi_m, phi_w, p['f1270_m'], p['f1270_w'],
                    p['f1270_c'], p['f1270_t'],
                    getattr(pwa, pref+'phif2_kk'), getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'))
            d4 = pwa.calculate_BW_BW(phi_m, phi_w, p['f2_m'], p['f2_w'],
                    p['f2_c'], p['f2_t'],
                    getattr(pwa, pref+'phif2_kk'), getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'))
        else:
            d1 = pwa.calculate_BW_flatte980(phi_m, phi_w, p['f980_m'], p['f980_g'], p['f980_r'],
                    p['f980_c'], p['f980_t'],
                    getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif0_kk'))
            d2 = pwa.calculate_BW_BW(phi_m, phi_w, p['f0_m'], p['f0_w'],
                    p['f0_c'], p['f0_t'],
                    getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif0_kk'))
            d3 = pwa.calculate_BW_flatte1270(phi_m, phi_w, p['f1270_m'], p['f1270_w'],
                    p['f1270_c'], p['f1270_t'],
                    getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif2_kk'))
            d4 = pwa.calculate_BW_BW(phi_m, phi_w, p['f2_m'], p['f2_w'],
                    p['f2_c'], p['f2_t'],
                    getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif2_kk'))
        return d1 + d2 + d3 + d4

    def test_data_likelihood_equivalence(self, pwa_funcs, test_args):
        """LLM data_likelihood must match jinja2 reference within 1e-10."""
        pwa_llm, pwa_j2 = pwa_funcs

        combined_llm = self._compute_combined(pwa_llm, test_args, is_llm=True, mc=False)
        combined_j2 = self._compute_combined(pwa_j2, test_args, is_llm=False, mc=False)

        lh_llm = -np.sum(np.log(np.sum(dplex.dabs(combined_llm), axis=1)))
        lh_j2 = -np.sum(np.log(np.sum(dplex.dabs(combined_j2), axis=1)))

        rel_diff = abs(lh_llm - lh_j2) / max(abs(lh_j2), 1e-15)
        assert rel_diff < 1e-10, f"Data lh differs: LLM={lh_llm:.15f}, j2={lh_j2:.15f}, rel_diff={rel_diff:.2e}"

    def test_mc_likelihood_equivalence(self, pwa_funcs, test_args):
        """LLM mc_likelihood must match jinja2 reference within 1e-10."""
        pwa_llm, pwa_j2 = pwa_funcs

        combined_llm = self._compute_combined(pwa_llm, test_args, is_llm=True, mc=True)
        combined_j2 = self._compute_combined(pwa_j2, test_args, is_llm=False, mc=True)

        mc_llm = np.sum(dplex.dabs(combined_llm))
        mc_j2 = np.sum(dplex.dabs(combined_j2))

        rel_diff = abs(mc_llm - mc_j2) / max(abs(mc_j2), 1e-15)
        assert rel_diff < 1e-10, f"MC lh differs: LLM={mc_llm:.15f}, j2={mc_j2:.15f}, rel_diff={rel_diff:.2e}"

    def test_gradient_equivalence(self, pwa_funcs, test_args):
        """LLM gradient must match jinja2 reference element-wise within 1e-9."""
        pwa_llm, pwa_j2 = pwa_funcs

        def data_lh_llm(a):
            combined = self._compute_combined(pwa_llm, a, is_llm=True, mc=False)
            return -np.sum(np.log(np.sum(dplex.dabs(combined), axis=1)))

        def data_lh_j2(a):
            combined = self._compute_combined(pwa_j2, a, is_llm=False, mc=False)
            return -np.sum(np.log(np.sum(dplex.dabs(combined), axis=1)))

        grad_llm = grad(data_lh_llm)(test_args)
        grad_j2 = grad(data_lh_j2)(test_args)

        max_diff = np.max(np.abs(grad_llm - grad_j2))
        assert max_diff < 1e-9, f"Gradient max diff {max_diff:.2e} exceeds 1e-9"

    def test_per_component_equivalence(self, pwa_funcs, test_args):
        """Each calculate_<prop> output must match individually."""
        pwa_llm, pwa_j2 = pwa_funcs
        phi_m, phi_w, p = self._make_params(test_args)

        components = [
            ("BW_flatte980", p['f980_m'], p['f980_g'], p['f980_r'], p['f980_c'], p['f980_t'],
             'data_phif0_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_flatte980, pwa_j2.calculate_BW_flatte980),
            ("BW_BW(phif0)", p['f0_m'], p['f0_w'], p['f0_c'], p['f0_t'],
             'data_phif0_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW),
            ("BW_flatte1270", p['f1270_m'], p['f1270_w'], p['f1270_c'], p['f1270_t'],
             'data_phif2_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_flatte1270, pwa_j2.calculate_BW_flatte1270),
            ("BW_BW(phif2)", p['f2_m'], p['f2_w'], p['f2_c'], p['f2_t'],
             'data_phif2_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW),
        ]

        for name, *params, amp_k, sbc_p_k, sbc_f_k, calc_llm, calc_j2 in components:
            amp_llm = getattr(pwa_llm, amp_k)
            sbc_p_llm = getattr(pwa_llm, sbc_p_k)
            sbc_f_llm = getattr(pwa_llm, sbc_f_k)
            amp_j2 = getattr(pwa_j2, amp_k)
            sbc_p_j2 = getattr(pwa_j2, sbc_p_k)
            sbc_f_j2 = getattr(pwa_j2, sbc_f_k)

            # LLM: (..., amp, Sbc_phi, Sbc_f)
            result_llm = calc_llm(phi_m, phi_w, *params, amp_llm, sbc_p_llm, sbc_f_llm)
            # jinja2: (..., Sbc_phi, Sbc_f, amp)
            result_j2 = calc_j2(phi_m, phi_w, *params, sbc_p_j2, sbc_f_j2, amp_j2)

            max_diff = np.max(np.abs(dplex.dabs(result_llm) - dplex.dabs(result_j2)))
            assert max_diff < 1e-10, f"{name}: max|dabs| diff = {max_diff:.2e}"


class TestNumericEquivalencePull:
    """Numeric equivalence tests for pull artifact (Sbc-first arg order)."""

    @pytest.fixture(scope="class")
    def pwa_funcs(self, mock_cdl):
        if not NUMERIC_READY:
            pytest.skip("JAX and dplex required for numeric tests")

        import importlib.util

        llm_path = "rendered_scripts/pull_object_kk.py"
        j2_path = "/tmp/pull_jinja2_ref.py"

        if not os.path.exists(llm_path):
            pytest.skip(f"{llm_path} not found")
        if not os.path.exists(j2_path):
            pytest.skip(f"{j2_path} not found")

        spec_llm = importlib.util.spec_from_file_location("pull_llm_numtest", llm_path)
        mod_llm = importlib.util.module_from_spec(spec_llm)
        sys.modules["pull_llm_numtest"] = mod_llm
        spec_llm.loader.exec_module(mod_llm)

        spec_j2 = importlib.util.spec_from_file_location("pull_j2_numtest", j2_path)
        mod_j2 = importlib.util.module_from_spec(spec_j2)
        sys.modules["pull_j2_numtest"] = mod_j2
        spec_j2.loader.exec_module(mod_j2)

        pwa_llm = mod_llm.PWAFunc(cdl=mock_cdl, device_id=None)
        pwa_j2 = mod_j2.PWAFunc(cdl=mock_cdl, device_id=None)
        return pwa_llm, pwa_j2

    def _make_params(self, args):
        phi_m = np.array([1.02])
        phi_w = np.array([0.004])
        p = {}
        p['f980_m'] = np.array([args[0]]); p['f980_g'] = np.array([args[1]]); p['f980_r'] = np.array([args[2]])
        p['f980_c'] = np.array([0.1, args[3]]).reshape(-1, 2)
        p['f980_t'] = np.array([0.1, args[4]]).reshape(-1, 2)
        p['f0_m'] = np.array([args[5]]); p['f0_w'] = np.array([args[6]])
        p['f0_c'] = np.array([args[7], args[8]]).reshape(-1, 2)
        p['f0_t'] = np.array([args[9], args[10]]).reshape(-1, 2)
        p['f1270_m'] = np.array([args[11]]); p['f1270_w'] = np.array([args[12]])
        p['f1270_c'] = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5)
        p['f1270_t'] = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5)
        p['f2_m'] = np.array([args[23], args[24], args[25]])
        p['f2_w'] = np.array([args[26], args[27], args[28]])
        p['f2_c'] = np.array([args[29], args[30], args[31], args[32], args[33], args[34],
                              args[35], args[36], args[37], args[38], args[39], args[40],
                              args[41], args[42], args[43]]).reshape(-1, 5)
        p['f2_t'] = np.array([args[44], args[45], args[46], args[47], args[48], args[49],
                              args[50], args[51], args[52], args[53], args[54], args[55],
                              args[56], args[57], args[58]]).reshape(-1, 5)
        return phi_m, phi_w, p

    def _compute_combined(self, pwa, args, mc=False):
        """Both LLM pull and jinja2 pull use Sbc-first order."""
        phi_m, phi_w, p = self._make_params(args)
        pref = "mc_" if mc else "data_"

        d1 = pwa.calculate_BW_flatte980(phi_m, phi_w, p['f980_m'], p['f980_g'], p['f980_r'],
                p['f980_c'], p['f980_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif0_kk'))
        d2 = pwa.calculate_BW_BW(phi_m, phi_w, p['f0_m'], p['f0_w'],
                p['f0_c'], p['f0_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif0_kk'))
        d3 = pwa.calculate_BW_flatte1270(phi_m, phi_w, p['f1270_m'], p['f1270_w'],
                p['f1270_c'], p['f1270_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif2_kk'))
        d4 = pwa.calculate_BW_BW(phi_m, phi_w, p['f2_m'], p['f2_w'],
                p['f2_c'], p['f2_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif2_kk'))
        return d1 + d2 + d3 + d4

    def test_data_likelihood_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        combined_llm = self._compute_combined(pwa_llm, test_args, mc=False)
        combined_j2 = self._compute_combined(pwa_j2, test_args, mc=False)
        lh_llm = -np.sum(np.log(np.sum(dplex.dabs(combined_llm), axis=1)))
        lh_j2 = -np.sum(np.log(np.sum(dplex.dabs(combined_j2), axis=1)))
        rel_diff = abs(lh_llm - lh_j2) / max(abs(lh_j2), 1e-15)
        assert rel_diff < 1e-10, f"Data lh differs: LLM={lh_llm:.15f}, j2={lh_j2:.15f}, rel_diff={rel_diff:.2e}"

    def test_mc_likelihood_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        combined_llm = self._compute_combined(pwa_llm, test_args, mc=True)
        combined_j2 = self._compute_combined(pwa_j2, test_args, mc=True)
        mc_llm = np.sum(dplex.dabs(combined_llm))
        mc_j2 = np.sum(dplex.dabs(combined_j2))
        rel_diff = abs(mc_llm - mc_j2) / max(abs(mc_j2), 1e-15)
        assert rel_diff < 1e-10, f"MC lh differs: LLM={mc_llm:.15f}, j2={mc_j2:.15f}, rel_diff={rel_diff:.2e}"

    def test_per_component_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        phi_m, phi_w, p = self._make_params(test_args)

        components = [
            ("BW_flatte980", p['f980_m'], p['f980_g'], p['f980_r'], p['f980_c'], p['f980_t'],
             'data_phif0_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_flatte980, pwa_j2.calculate_BW_flatte980),
            ("BW_BW(phif0)", p['f0_m'], p['f0_w'], p['f0_c'], p['f0_t'],
             'data_phif0_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW),
            ("BW_flatte1270", p['f1270_m'], p['f1270_w'], p['f1270_c'], p['f1270_t'],
             'data_phif2_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_flatte1270, pwa_j2.calculate_BW_flatte1270),
            ("BW_BW(phif2)", p['f2_m'], p['f2_w'], p['f2_c'], p['f2_t'],
             'data_phif2_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW),
        ]

        for name, *params, amp_k, sbc_p_k, sbc_f_k, calc_llm, calc_j2 in components:
            amp_llm = getattr(pwa_llm, amp_k)
            sbc_p_llm = getattr(pwa_llm, sbc_p_k)
            sbc_f_llm = getattr(pwa_llm, sbc_f_k)
            amp_j2 = getattr(pwa_j2, amp_k)
            sbc_p_j2 = getattr(pwa_j2, sbc_p_k)
            sbc_f_j2 = getattr(pwa_j2, sbc_f_k)

            result_llm = calc_llm(phi_m, phi_w, *params, sbc_p_llm, sbc_f_llm, amp_llm)
            result_j2 = calc_j2(phi_m, phi_w, *params, sbc_p_j2, sbc_f_j2, amp_j2)

            max_diff = np.max(np.abs(dplex.dabs(result_llm) - dplex.dabs(result_j2)))
            assert max_diff < 1e-10, f"{name}: max|dabs| diff = {max_diff:.2e}"


class TestNumericEquivalenceDrawLH:
    """Numeric equivalence tests for draw_lh artifact (Sbc-first arg order)."""

    @pytest.fixture(scope="class")
    def pwa_funcs(self, mock_cdl):
        if not NUMERIC_READY:
            pytest.skip("JAX and dplex required for numeric tests")

        import importlib.util

        llm_path = "rendered_scripts/draw_lh_object_kk.py"
        j2_path = "/tmp/draw_lh_jinja2_ref.py"

        if not os.path.exists(llm_path):
            pytest.skip(f"{llm_path} not found")
        if not os.path.exists(j2_path):
            pytest.skip(f"{j2_path} not found")

        spec_llm = importlib.util.spec_from_file_location("drawlh_llm_numtest", llm_path)
        mod_llm = importlib.util.module_from_spec(spec_llm)
        sys.modules["drawlh_llm_numtest"] = mod_llm
        spec_llm.loader.exec_module(mod_llm)

        spec_j2 = importlib.util.spec_from_file_location("drawlh_j2_numtest", j2_path)
        mod_j2 = importlib.util.module_from_spec(spec_j2)
        sys.modules["drawlh_j2_numtest"] = mod_j2
        spec_j2.loader.exec_module(mod_j2)

        pwa_llm = mod_llm.PWAFunc(cdl=mock_cdl, device_id=None)
        pwa_j2 = mod_j2.PWAFunc(cdl=mock_cdl, device_id=None)
        return pwa_llm, pwa_j2

    def _make_params(self, args):
        phi_m = np.array([1.02])
        phi_w = np.array([0.004])
        p = {}
        p['f980_m'] = np.array([args[0]]); p['f980_g'] = np.array([args[1]]); p['f980_r'] = np.array([args[2]])
        p['f980_c'] = np.array([0.1, args[3]]).reshape(-1, 2)
        p['f980_t'] = np.array([0.1, args[4]]).reshape(-1, 2)
        p['f0_m'] = np.array([args[5]]); p['f0_w'] = np.array([args[6]])
        p['f0_c'] = np.array([args[7], args[8]]).reshape(-1, 2)
        p['f0_t'] = np.array([args[9], args[10]]).reshape(-1, 2)
        p['f1270_m'] = np.array([args[11]]); p['f1270_w'] = np.array([args[12]])
        p['f1270_c'] = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5)
        p['f1270_t'] = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5)
        p['f2_m'] = np.array([args[23], args[24], args[25]])
        p['f2_w'] = np.array([args[26], args[27], args[28]])
        p['f2_c'] = np.array([args[29], args[30], args[31], args[32], args[33], args[34],
                              args[35], args[36], args[37], args[38], args[39], args[40],
                              args[41], args[42], args[43]]).reshape(-1, 5)
        p['f2_t'] = np.array([args[44], args[45], args[46], args[47], args[48], args[49],
                              args[50], args[51], args[52], args[53], args[54], args[55],
                              args[56], args[57], args[58]]).reshape(-1, 5)
        return phi_m, phi_w, p

    def _compute_combined(self, pwa, args, mc=False):
        phi_m, phi_w, p = self._make_params(args)
        pref = "mc_" if mc else "data_"

        d1 = pwa.calculate_BW_flatte980(phi_m, phi_w, p['f980_m'], p['f980_g'], p['f980_r'],
                p['f980_c'], p['f980_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif0_kk'))
        d2 = pwa.calculate_BW_BW(phi_m, phi_w, p['f0_m'], p['f0_w'],
                p['f0_c'], p['f0_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif0_kk'))
        d3 = pwa.calculate_BW_flatte1270(phi_m, phi_w, p['f1270_m'], p['f1270_w'],
                p['f1270_c'], p['f1270_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif2_kk'))
        d4 = pwa.calculate_BW_BW(phi_m, phi_w, p['f2_m'], p['f2_w'],
                p['f2_c'], p['f2_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif2_kk'))
        return d1 + d2 + d3 + d4

    def test_data_likelihood_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        combined_llm = self._compute_combined(pwa_llm, test_args, mc=False)
        combined_j2 = self._compute_combined(pwa_j2, test_args, mc=False)
        lh_llm = -np.sum(np.log(np.sum(dplex.dabs(combined_llm), axis=1)))
        lh_j2 = -np.sum(np.log(np.sum(dplex.dabs(combined_j2), axis=1)))
        rel_diff = abs(lh_llm - lh_j2) / max(abs(lh_j2), 1e-15)
        assert rel_diff < 1e-10, f"Data lh differs: LLM={lh_llm:.15f}, j2={lh_j2:.15f}, rel_diff={rel_diff:.2e}"

    def test_mc_likelihood_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        combined_llm = self._compute_combined(pwa_llm, test_args, mc=True)
        combined_j2 = self._compute_combined(pwa_j2, test_args, mc=True)
        mc_llm = np.sum(dplex.dabs(combined_llm))
        mc_j2 = np.sum(dplex.dabs(combined_j2))
        rel_diff = abs(mc_llm - mc_j2) / max(abs(mc_j2), 1e-15)
        assert rel_diff < 1e-10, f"MC lh differs: LLM={mc_llm:.15f}, j2={mc_j2:.15f}, rel_diff={rel_diff:.2e}"

    def test_per_component_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        phi_m, phi_w, p = self._make_params(test_args)

        components = [
            ("BW_flatte980", p['f980_m'], p['f980_g'], p['f980_r'], p['f980_c'], p['f980_t'],
             'data_phif0_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_flatte980, pwa_j2.calculate_BW_flatte980),
            ("BW_BW(phif0)", p['f0_m'], p['f0_w'], p['f0_c'], p['f0_t'],
             'data_phif0_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW),
            ("BW_flatte1270", p['f1270_m'], p['f1270_w'], p['f1270_c'], p['f1270_t'],
             'data_phif2_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_flatte1270, pwa_j2.calculate_BW_flatte1270),
            ("BW_BW(phif2)", p['f2_m'], p['f2_w'], p['f2_c'], p['f2_t'],
             'data_phif2_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW),
        ]

        for name, *params, amp_k, sbc_p_k, sbc_f_k, calc_llm, calc_j2 in components:
            amp_llm = getattr(pwa_llm, amp_k)
            sbc_p_llm = getattr(pwa_llm, sbc_p_k)
            sbc_f_llm = getattr(pwa_llm, sbc_f_k)
            amp_j2 = getattr(pwa_j2, amp_k)
            sbc_p_j2 = getattr(pwa_j2, sbc_p_k)
            sbc_f_j2 = getattr(pwa_j2, sbc_f_k)

            result_llm = calc_llm(phi_m, phi_w, *params, sbc_p_llm, sbc_f_llm, amp_llm)
            result_j2 = calc_j2(phi_m, phi_w, *params, sbc_p_j2, sbc_f_j2, amp_j2)

            max_diff = np.max(np.abs(dplex.dabs(result_llm) - dplex.dabs(result_j2)))
            assert max_diff < 1e-10, f"{name}: max|dabs| diff = {max_diff:.2e}"


class TestNumericEquivalenceLasso:
    """Numeric equivalence tests for lasso artifact (Sbc-first arg order)."""

    @pytest.fixture(scope="class")
    def pwa_funcs(self, mock_cdl):
        if not NUMERIC_READY:
            pytest.skip("JAX and dplex required for numeric tests")

        import importlib.util

        llm_path = "rendered_scripts/lasso_object_kk.py"
        j2_path = "/tmp/lasso_jinja2_ref.py"

        if not os.path.exists(llm_path):
            pytest.skip(f"{llm_path} not found")
        if not os.path.exists(j2_path):
            pytest.skip(f"{j2_path} not found")

        sys.path.insert(0, "/tmp")
        import ROOT  # noqa: F401 — mock ROOT for jinja2 reference

        spec_llm = importlib.util.spec_from_file_location("lasso_llm_numtest", llm_path)
        mod_llm = importlib.util.module_from_spec(spec_llm)
        sys.modules["lasso_llm_numtest"] = mod_llm
        spec_llm.loader.exec_module(mod_llm)

        spec_j2 = importlib.util.spec_from_file_location("lasso_j2_numtest", j2_path)
        mod_j2 = importlib.util.module_from_spec(spec_j2)
        sys.modules["lasso_j2_numtest"] = mod_j2
        spec_j2.loader.exec_module(mod_j2)

        pwa_llm = mod_llm.PWAFunc(cdl=mock_cdl, device_id=None)
        pwa_j2 = mod_j2.PWAFunc(cdl=mock_cdl, device_id=None)

        # Set truth data for lasso (not loaded from CDL by default in the jinja2 reference)
        for attr in ['truth_phi_kk', 'truth_f_kk', 'truth_phif0_kk', 'truth_phif2_kk']:
            for pwa in [pwa_llm, pwa_j2]:
                if getattr(pwa, attr, None) is None and hasattr(mock_cdl, attr):
                    setattr(pwa, attr, getattr(mock_cdl, attr))

        return pwa_llm, pwa_j2

    def _make_params(self, args):
        phi_m = np.array([1.02])
        phi_w = np.array([0.004])
        p = {}
        p['f980_m'] = np.array([args[0]]); p['f980_g'] = np.array([args[1]]); p['f980_r'] = np.array([args[2]])
        p['f980_c'] = np.array([0.1, args[3]]).reshape(-1, 2)
        p['f980_t'] = np.array([0.1, args[4]]).reshape(-1, 2)
        p['f0_m'] = np.array([args[5]]); p['f0_w'] = np.array([args[6]])
        p['f0_c'] = np.array([args[7], args[8]]).reshape(-1, 2)
        p['f0_t'] = np.array([args[9], args[10]]).reshape(-1, 2)
        p['f1270_m'] = np.array([args[11]]); p['f1270_w'] = np.array([args[12]])
        p['f1270_c'] = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5)
        p['f1270_t'] = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5)
        p['f2_m'] = np.array([args[23], args[24], args[25]])
        p['f2_w'] = np.array([args[26], args[27], args[28]])
        p['f2_c'] = np.array([args[29], args[30], args[31], args[32], args[33], args[34],
                              args[35], args[36], args[37], args[38], args[39], args[40],
                              args[41], args[42], args[43]]).reshape(-1, 5)
        p['f2_t'] = np.array([args[44], args[45], args[46], args[47], args[48], args[49],
                              args[50], args[51], args[52], args[53], args[54], args[55],
                              args[56], args[57], args[58]]).reshape(-1, 5)
        return phi_m, phi_w, p

    def _compute_combined(self, pwa, args, mc=False):
        phi_m, phi_w, p = self._make_params(args)
        pref = "mc_" if mc else "data_"

        d1 = pwa.calculate_BW_flatte980(phi_m, phi_w, p['f980_m'], p['f980_g'], p['f980_r'],
                p['f980_c'], p['f980_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif0_kk'))
        d2 = pwa.calculate_BW_BW(phi_m, phi_w, p['f0_m'], p['f0_w'],
                p['f0_c'], p['f0_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif0_kk'))
        d3 = pwa.calculate_BW_flatte1270(phi_m, phi_w, p['f1270_m'], p['f1270_w'],
                p['f1270_c'], p['f1270_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif2_kk'))
        d4 = pwa.calculate_BW_BW(phi_m, phi_w, p['f2_m'], p['f2_w'],
                p['f2_c'], p['f2_t'],
                getattr(pwa, pref+'phi_kk'), getattr(pwa, pref+'f_kk'), getattr(pwa, pref+'phif2_kk'))
        return d1 + d2 + d3 + d4

    def test_data_likelihood_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        combined_llm = self._compute_combined(pwa_llm, test_args, mc=False)
        combined_j2 = self._compute_combined(pwa_j2, test_args, mc=False)
        lh_llm = -np.sum(np.log(np.sum(dplex.dabs(combined_llm), axis=1)))
        lh_j2 = -np.sum(np.log(np.sum(dplex.dabs(combined_j2), axis=1)))
        rel_diff = abs(lh_llm - lh_j2) / max(abs(lh_j2), 1e-15)
        assert rel_diff < 1e-10, f"Data lh differs: LLM={lh_llm:.15f}, j2={lh_j2:.15f}, rel_diff={rel_diff:.2e}"

    def test_mc_likelihood_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        combined_llm = self._compute_combined(pwa_llm, test_args, mc=True)
        combined_j2 = self._compute_combined(pwa_j2, test_args, mc=True)
        mc_llm = np.sum(dplex.dabs(combined_llm))
        mc_j2 = np.sum(dplex.dabs(combined_j2))
        rel_diff = abs(mc_llm - mc_j2) / max(abs(mc_j2), 1e-15)
        assert rel_diff < 1e-10, f"MC lh differs: LLM={mc_llm:.15f}, j2={mc_j2:.15f}, rel_diff={rel_diff:.2e}"

    def test_per_component_equivalence(self, pwa_funcs, test_args):
        pwa_llm, pwa_j2 = pwa_funcs
        phi_m, phi_w, p = self._make_params(test_args)

        components = [
            ("BW_flatte980", p['f980_m'], p['f980_g'], p['f980_r'], p['f980_c'], p['f980_t'],
             'data_phif0_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_flatte980, pwa_j2.calculate_BW_flatte980),
            ("BW_BW(phif0)", p['f0_m'], p['f0_w'], p['f0_c'], p['f0_t'],
             'data_phif0_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW),
            ("BW_flatte1270", p['f1270_m'], p['f1270_w'], p['f1270_c'], p['f1270_t'],
             'data_phif2_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_flatte1270, pwa_j2.calculate_BW_flatte1270),
            ("BW_BW(phif2)", p['f2_m'], p['f2_w'], p['f2_c'], p['f2_t'],
             'data_phif2_kk', 'data_phi_kk', 'data_f_kk',
             pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW),
        ]

        for name, *params, amp_k, sbc_p_k, sbc_f_k, calc_llm, calc_j2 in components:
            amp_llm = getattr(pwa_llm, amp_k)
            sbc_p_llm = getattr(pwa_llm, sbc_p_k)
            sbc_f_llm = getattr(pwa_llm, sbc_f_k)
            amp_j2 = getattr(pwa_j2, amp_k)
            sbc_p_j2 = getattr(pwa_j2, sbc_p_k)
            sbc_f_j2 = getattr(pwa_j2, sbc_f_k)

            result_llm = calc_llm(phi_m, phi_w, *params, sbc_p_llm, sbc_f_llm, amp_llm)
            result_j2 = calc_j2(phi_m, phi_w, *params, sbc_p_j2, sbc_f_j2, amp_j2)

            max_diff = np.max(np.abs(dplex.dabs(result_llm) - dplex.dabs(result_j2)))
            assert max_diff < 1e-10, f"{name}: max|dabs| diff = {max_diff:.2e}"

    def test_lasso_calculate_equivalence(self, pwa_funcs, test_args):
        """lasso_calculate_* functions must match between LLM and jinja2."""
        pwa_llm, pwa_j2 = pwa_funcs
        phi_m, phi_w, p = self._make_params(test_args)

        lasso_components = [
            ("lasso_BW_flatte980", p['f980_m'], p['f980_g'], p['f980_r'], p['f980_c'], p['f980_t'],
             'truth_phif0_kk', 'truth_phi_kk', 'truth_f_kk',
             pwa_llm.lasso_calculate_BW_flatte980, pwa_j2.lasso_calculate_BW_flatte980),
            ("lasso_BW_BW(phif0)", p['f0_m'], p['f0_w'], p['f0_c'], p['f0_t'],
             'truth_phif0_kk', 'truth_phi_kk', 'truth_f_kk',
             pwa_llm.lasso_calculate_BW_BW, pwa_j2.lasso_calculate_BW_BW),
            ("lasso_BW_flatte1270", p['f1270_m'], p['f1270_w'], p['f1270_c'], p['f1270_t'],
             'truth_phif2_kk', 'truth_phi_kk', 'truth_f_kk',
             pwa_llm.lasso_calculate_BW_flatte1270, pwa_j2.lasso_calculate_BW_flatte1270),
            ("lasso_BW_BW(phif2)", p['f2_m'], p['f2_w'], p['f2_c'], p['f2_t'],
             'truth_phif2_kk', 'truth_phi_kk', 'truth_f_kk',
             pwa_llm.lasso_calculate_BW_BW, pwa_j2.lasso_calculate_BW_BW),
        ]

        for name, *params, amp_k, sbc_p_k, sbc_f_k, calc_llm, calc_j2 in lasso_components:
            amp_llm = getattr(pwa_llm, amp_k)
            sbc_p_llm = getattr(pwa_llm, sbc_p_k)
            sbc_f_llm = getattr(pwa_llm, sbc_f_k)
            amp_j2 = getattr(pwa_j2, amp_k)
            sbc_p_j2 = getattr(pwa_j2, sbc_p_k)
            sbc_f_j2 = getattr(pwa_j2, sbc_f_k)

            result_llm = calc_llm(phi_m, phi_w, *params, sbc_p_llm, sbc_f_llm, amp_llm)
            result_j2 = calc_j2(phi_m, phi_w, *params, sbc_p_j2, sbc_f_j2, amp_j2)

            max_diff = np.max(np.abs(dplex.dabs(result_llm) - dplex.dabs(result_j2)))
            assert max_diff < 1e-10, f"{name}: max|dabs| diff = {max_diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
