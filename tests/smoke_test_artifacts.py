#!/usr/bin/env python3
"""Quick smoke test: import pull/draw_lh/lasso LLM + jinja2, check calculate outputs match."""
import os, sys
import numpy as onp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/tmp")  # for ROOT mock

import jax.numpy as np
from dlib import dplex

n_data, n_mc = 100, 1000

class MockCDL:
    def __init__(self):
        rng = onp.random.RandomState(42)
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

cdl = MockCDL()

test_args = onp.array([
    0.99, 0.05, 1.0, 0.5, 0.3, 1.7, 0.2, 0.4, -0.2, 0.6,
    0.8, 1.27, 0.15, 0.3, -0.1, 0.2, -0.3, 0.4, 0.1, 0.2,
    0.3, 0.4, 0.5, 1.5, 2.1, 2.3, 0.1, 0.15, 0.2, 0.3,
    -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7, 0.8,
    -0.8, 0.9, -0.9, 1.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8
], dtype=onp.float64)

import importlib.util

for artifact in ['pull', 'draw_lh', 'lasso']:
    print(f"\n{'='*60}")
    print(f"Testing {artifact}")
    print(f"{'='*60}")

    llm_path = f"rendered_scripts/{artifact}_object_kk.py"
    j2_path = f"/tmp/{artifact}_jinja2_ref.py"

    # Import LLM
    name_llm = f"{artifact}_llm_test"
    spec_llm = importlib.util.spec_from_file_location(name_llm, llm_path)
    mod_llm = importlib.util.module_from_spec(spec_llm)
    sys.modules[name_llm] = mod_llm
    spec_llm.loader.exec_module(mod_llm)

    # Import jinja2 (lasso needs ROOT mock at /tmp)
    j2_available = True
    try:
        name_j2 = f"{artifact}_j2_test"
        spec_j2 = importlib.util.spec_from_file_location(name_j2, j2_path)
        mod_j2 = importlib.util.module_from_spec(spec_j2)
        sys.modules[name_j2] = mod_j2
        spec_j2.loader.exec_module(mod_j2)
    except (ModuleNotFoundError, ImportError) as e:
        print(f"  Jinja2 reference not available: {e}")
        j2_available = False

    if not j2_available:
        print(f"  Skipping comparison tests for {artifact}")
        continue

    pwa_llm = mod_llm.PWAFunc(cdl=cdl, device_id=None)
    pwa_j2 = mod_j2.PWAFunc(cdl=cdl, device_id=None)

    # Ensure truth data is available for lasso (loaded from CDL if not from files)
    for attr in ['truth_phi_kk', 'truth_f_kk', 'truth_phif0_kk', 'truth_phif2_kk']:
        for pwa in [pwa_llm, pwa_j2]:
            if getattr(pwa, attr, None) is None and hasattr(cdl, attr):
                setattr(pwa, attr, getattr(cdl, attr))

    phi_m = np.array([1.02]); phi_w = np.array([0.004])

    kk_f980_m = np.array([test_args[0]]); kk_f980_g = np.array([test_args[1]]); kk_f980_r = np.array([test_args[2]])
    kk_f980_c = np.array([0.1, test_args[3]]).reshape(-1,2)
    kk_f980_t = np.array([0.1, test_args[4]]).reshape(-1,2)

    kk_f0_m = np.array([test_args[5]]); kk_f0_w = np.array([test_args[6]])
    kk_f0_c = np.array([test_args[7], test_args[8]]).reshape(-1,2)
    kk_f0_t = np.array([test_args[9], test_args[10]]).reshape(-1,2)

    kk_f1270_m = np.array([test_args[11]]); kk_f1270_w = np.array([test_args[12]])
    kk_f1270_c = np.array([test_args[13], test_args[14], test_args[15], test_args[16], test_args[17]]).reshape(-1,5)
    kk_f1270_t = np.array([test_args[18], test_args[19], test_args[20], test_args[21], test_args[22]]).reshape(-1,5)

    kk_f2_m = np.array([test_args[23], test_args[24], test_args[25]])
    kk_f2_w = np.array([test_args[26], test_args[27], test_args[28]])
    kk_f2_c = np.array([test_args[29], test_args[30], test_args[31], test_args[32], test_args[33], test_args[34],
                       test_args[35], test_args[36], test_args[37], test_args[38], test_args[39], test_args[40],
                       test_args[41], test_args[42], test_args[43]]).reshape(-1,5)
    kk_f2_t = np.array([test_args[44], test_args[45], test_args[46], test_args[47], test_args[48], test_args[49],
                       test_args[50], test_args[51], test_args[52], test_args[53], test_args[54], test_args[55],
                       test_args[56], test_args[57], test_args[58]]).reshape(-1,5)

    # Test each calculate function with correct args
    # Jinja2 pull/draw_lh/lasso use Sbc-first order: (..., Sbc_phi, Sbc_f, amp)
    # LLM pull/draw_lh also use Sbc-first (same order as jinja2 for these artifacts)

    # BW_flatte980 args: phi_m, phi_w, f980_m, f980_g, f980_r, f980_c, f980_t, Sbc_phi, Sbc_f, amp
    bf980_args = [phi_m, phi_w, kk_f980_m, kk_f980_g, kk_f980_r, kk_f980_c, kk_f980_t]
    # BW_BW args: phi_m, phi_w, f0_m, f0_w, f0_c, f0_t, Sbc_phi, Sbc_f, amp
    bbw_args = [phi_m, phi_w, kk_f0_m, kk_f0_w, kk_f0_c, kk_f0_t]

    results = {}
    for name, calc_llm, calc_j2, extra_args in [
        ("BW_flatte980", pwa_llm.calculate_BW_flatte980, pwa_j2.calculate_BW_flatte980, bf980_args),
        ("BW_BW", pwa_llm.calculate_BW_BW, pwa_j2.calculate_BW_BW, bbw_args),
    ]:
        try:
            r_llm = calc_llm(*extra_args, pwa_llm.data_phi_kk, pwa_llm.data_f_kk, pwa_llm.data_phif0_kk)
            r_j2 = calc_j2(*extra_args, pwa_j2.data_phi_kk, pwa_j2.data_f_kk, pwa_j2.data_phif0_kk)
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            continue

        d_llm = dplex.dabs(r_llm)
        d_j2 = dplex.dabs(r_j2)
        max_diff = np.max(np.abs(d_llm - d_j2))
        status = "PASS" if max_diff < 1e-10 else f"FAIL (diff={max_diff:.2e})"
        print(f"  {name}: shape={r_llm.shape} {status}")

    # Try to call data_likelihood_kk
    try:
        lh_llm = pwa_llm.data_likelihood_kk(test_args)
        lh_j2 = pwa_j2.data_likelihood_kk(test_args)
        diff = abs(lh_llm - lh_j2)
        rel = diff / max(abs(lh_j2), 1e-15)
        status = "PASS" if rel < 1e-10 else f"FAIL (rel={rel:.2e})"
        print(f"  data_likelihood: LLM={lh_llm:.6f}, j2={lh_j2:.6f} {status}")
    except Exception as e:
        print(f"  data_likelihood: ERROR - {e}")

print("\nDone!")
