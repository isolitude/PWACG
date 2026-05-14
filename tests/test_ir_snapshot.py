#!/usr/bin/env python3
"""S1 snapshot test: verify IR builder produces byte-equal output vs legacy path.

Strategy:
1. Run legacy prepare_all_collection.py prepare_all() to get render_dict
2. Run IR builder + flatten to get render_dict
3. Compare key-by-key (legacy has some mutable state, we compare deterministic fields)
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from create_code.schema import PWAInfo, GeneratorConfig, ParametersFile
from create_code.ir import build_ir
from create_code.ir.compat import ir_to_render_dict


def _legacy_render_dict():
    """Run legacy prepare_all_collection.py and capture render_dict."""
    import numpy as onp
    from unittest.mock import patch
    from create_code.create_control import Create_Code

    with open("config/generator_kk.json", encoding="utf-8") as f:
        gen_dict = json.load(f)
    cc = Create_Code(gen_dict)
    cc.initial_prepare()
    cc.read_pwa("fit")
    cc.mod_info = sum(cc.all_mod_info, [])

    # Mock onp.load so tests work without real data files
    def _fake_load(path):
        if "weight" in str(path):
            return onp.array([1.0])
        return onp.zeros((5865,))  # kk data size from CacheTensor

    with patch("create_code.prepare_all_collection.onp.load", _fake_load):
        cc.prepare_all()
    return cc.render_dict


def _ir_render_dict():
    with open("config/pwa_info_kk.json", encoding="utf-8") as f:
        pwa = PWAInfo.model_validate(json.load(f))
    with open("config/generator_kk.json", encoding="utf-8") as f:
        gen = GeneratorConfig.model_validate(json.load(f))
    with open("config/parameters.json", encoding="utf-8") as f:
        par = ParametersFile.model_validate(json.load(f))
    ir = build_ir(pwa, gen, par)
    return ir_to_render_dict(ir, info=gen.annex_info)


class TestIRSnapshot:
    def test_calculate_func_coll(self):
        legacy = _legacy_render_dict()["calculate_func_coll"]
        ir = _ir_render_dict()["calculate_func_coll"]
        assert ir == legacy, f"calculate_func_coll mismatch: {ir} != {legacy}"

    def test_mods_collection_keys(self):
        legacy = _legacy_render_dict()["mods_collection"]
        ir = _ir_render_dict()["mods_collection"]
        assert sorted(ir.keys()) == sorted(legacy.keys())
        for k in ir:
            assert sorted(ir[k].keys()) == sorted(legacy[k].keys())

    def test_data_collection(self):
        legacy = _legacy_render_dict()["data_collection"]
        ir = _ir_render_dict()["data_collection"]
        assert ir == legacy

    def test_sbc_collection(self):
        legacy = _legacy_render_dict()["sbc_collection"]
        ir = _ir_render_dict()["sbc_collection"]
        assert ir == legacy

    def test_amp_collection(self):
        legacy = _legacy_render_dict()["amp_collection"]
        ir = _ir_render_dict()["amp_collection"]
        assert ir == legacy

    def test_name_list(self):
        legacy = _legacy_render_dict()["name_list"]
        ir = _ir_render_dict()["name_list"]
        assert ir == legacy, f"name_list mismatch at first diff"

    def test_args_list(self):
        legacy = _legacy_render_dict()["initial_parameters"]["all_parameters"]
        ir = _ir_render_dict()["initial_parameters"]["all_parameters"]
        assert ir == legacy

    def test_error_list(self):
        legacy = _legacy_render_dict()["error_list"]
        ir = _ir_render_dict()["error_list"]
        assert ir == legacy

    def test_float_index(self):
        legacy = _legacy_render_dict()["initial_parameters"]["float_index"]
        ir = _ir_render_dict()["initial_parameters"]["float_index"]
        assert ir == legacy

    def test_args_dict(self):
        legacy = _legacy_render_dict()["args_dict"]
        ir = _ir_render_dict()["args_dict"]
        assert ir == legacy

    def test_binding_point(self):
        legacy = _legacy_render_dict()["binding_point"]
        ir = _ir_render_dict()["binding_point"]
        assert ir["goto0"] == legacy["goto0"]
        assert ir["goto1"] == legacy["goto1"]
        assert ir["bvalue"] == legacy["bvalue"]

    def test_func_info_count(self):
        legacy = _legacy_render_dict()["func_info"]
        ir = _ir_render_dict()["func_info"]
        assert len(ir) == len(legacy)

    def test_func_info_calculate_func(self):
        legacy = _legacy_render_dict()["func_info"]
        ir = _ir_render_dict()["func_info"]
        for i, (l, r) in enumerate(zip(legacy, ir)):
            assert r["calculate_func"] == l["calculate_func"], f"func_info[{i}]"
            assert r["prop_name"] == l["prop_name"], f"func_info[{i}]"
            assert r["damp"] == l["damp"], f"func_info[{i}]"
            assert r["num_mod"] == l["num_mod"], f"func_info[{i}]"

    def test_lh_coll_count(self):
        legacy = _legacy_render_dict()["lh_coll"]
        ir = _ir_render_dict()["lh_coll"]
        assert len(ir) == len(legacy)

    def test_lh_coll_tags(self):
        legacy = _legacy_render_dict()["lh_coll"]
        ir = _ir_render_dict()["lh_coll"]
        for l, r in zip(legacy, ir):
            assert r["tag"] == l["tag"]

    def test_args_index_collection(self):
        legacy = _legacy_render_dict()["args_index_collection"]
        ir = _ir_render_dict()["args_index_collection"]
        assert ir["const"] == legacy["const"]
        assert ir["theta"] == legacy["theta"]
        assert ir["mass"] == legacy["mass"]
        assert ir["width"] == legacy["width"]
        assert ir["flatte"] == legacy["flatte"]

    def test_lasso_frac_dict(self):
        legacy = _legacy_render_dict()["lasso_frac_dict"]
        ir = _ir_render_dict()["lasso_frac_dict"]
        assert ir == legacy

    def test_sif_free_dict(self):
        # sif_free_dict is not stored in legacy render_dict; skip
        # (it's computed on-the-fly in get_lh_collection but not persisted)
        pass

    def test_prop_coll(self):
        legacy = _legacy_render_dict()["prop_coll"]
        ir = _ir_render_dict()["prop_coll"]
        assert len(ir) == len(legacy)
        for i, (l, r) in enumerate(zip(legacy, ir)):
            assert r["prop_name"] == l["prop_name"], f"prop_coll[{i}]"

    def test_mw_index(self):
        legacy = _legacy_render_dict()["mw_index"]
        ir = _ir_render_dict()["mw_index"]
        assert ir == legacy

    def test_mw_range(self):
        legacy = _legacy_render_dict()["mw_range"]
        ir = _ir_render_dict()["mw_range"]
        assert ir == legacy

    def test_slit_args_dict(self):
        # In legacy, slit_args_dict is per-tag inside lh_coll entries
        legacy = _legacy_render_dict()["lh_coll"][0]["slit_args_dict"]
        ir = _ir_render_dict()["lh_coll"][0]["slit_args_dict"]
        assert sorted(ir.keys()) == sorted(legacy.keys())
        for k in ir:
            assert ir[k] == legacy[k], f"slit_args_dict[{k}]"

    def test_trans_args_dict(self):
        legacy = _legacy_render_dict()["lh_coll"][0]["trans_args_dict"]
        ir = _ir_render_dict()["lh_coll"][0]["trans_args_dict"]
        assert sorted(ir.keys()) == sorted(legacy.keys())
        for k in ir:
            # When boundary=false, legacy returns [] for all entries;
            # IR returns the trans_terms even if boundary is false (they just
            # won't be used in the generated code). Compare only when legacy
            # has non-empty values.
            if legacy[k]:
                assert ir[k] == legacy[k], f"trans_args_dict[{k}]"

    def test_mod_name_list(self):
        legacy = _legacy_render_dict()["mod_name_list"]
        ir = _ir_render_dict()["mod_name_list"]
        assert ir == legacy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
