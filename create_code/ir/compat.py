#!/usr/bin/env python3
# coding: utf-8
"""Compatibility shim: flatten PWAIR back into the legacy render_dict.

Used during S1 to verify IR correctness: build IR, flatten to render_dict,
feed to existing jinja2 templates, compare output byte-equal with legacy path.

This module is temporary — deleted in S5 when jinja2 is removed.
"""
from __future__ import annotations
from typing import Any

from .ir_models import PWAIR, FuncInfo, SlitArg


def ir_to_render_dict(ir: PWAIR, *, info: Any) -> dict[str, Any]:
    """Convert PWAIR to the legacy render_dict expected by jinja2 templates."""
    rd: dict[str, Any] = {
        "generator_id": ir.generator_id,
        "info": info,
        "mod_info": [],  # will populate from mods_collection
        "calculate_func_coll": list(ir.calculate_func_coll),
        "mods_collection": ir.mods_collection,
        "func_info": [],
        "data_collection": list(ir.data_collection),
        "sbc_collection": list(ir.sbc_collection),
        "amp_collection": list(ir.amp_collection),
        "args_collection": {},
        "name_list": list(ir.params.name_list),
        "error_list": list(ir.params.error_list),
        "args_dict": {k: list(v) for k, v in ir.args_dict.items()},
        "all_const_index": list(ir.all_const_index),
        "binding_point": {
            "goto0": list(ir.binding.goto0),
            "goto1": list(ir.binding.goto1),
            "bvalue": list(ir.binding.bvalue),
        },
        "initial_parameters": {
            "all_parameters": ir.initial_parameters["all_parameters"],
            "float_index": ir.initial_parameters["float_index"],
        },
        "slit_args_dict": {k: v.expr for k, v in ir.slit_args.items()},
        "trans_args_dict": {k: list(v.trans_terms) for k, v in ir.slit_args.items()},
        "lh_coll": [],
        "args_index_collection": {
            "const": list(ir.args_index_collection.const),
            "theta": list(ir.args_index_collection.theta),
            "mass": list(ir.args_index_collection.mass),
            "width": list(ir.args_index_collection.width),
            "flatte": list(ir.args_index_collection.flatte),
        },
        "lasso_frac_dict": {k: list(v) for k, v in ir.lasso.lasso_frac_dict.items()},
        "sif_free_dict": {k: list(v) for k, v in ir.lasso.sif_free_dict.items()},
        "mod_name_list": list(ir.lasso.mod_name_list),
        "prop_coll": [],
        "name_index_complete": ir.params.name_to_index,
        "range_dict": {},
        "mw_index": list(ir.mw_index),
        "mw_range": [list(r) for r in ir.mw_range],
    }

    # func_info (with legacy additions)
    for f in ir.funcs:
        fd = {
            "calculate_func": f.calculate_func,
            "prop_name": f.prop_name,
            "amp": f.amp,
            "Sbc": f.Sbc,
            "prop": f.prop,
            "all_paras": list(f.all_paras),
            "compl_paras": list(f.compl_paras),
            "theta": f.theta,
            "const": f.const,
            "damp": f.damp,
            "num_mod": f.num_mod,
            "const_index": list(f.const_index),
            "mod_name_list": [dict(d) for d in f.mod_name_list],
        }
        if f.theta:
            fd[f.theta] = f.theta
        if f.const:
            fd[f.const] = f.const
        # merge paras
        for mp in f.merge_paras:
            fd.setdefault(mp, [])
            if mp not in fd:
                fd[mp] = []
        rd["func_info"].append(fd)

    # prop_coll (first-seen dedup)
    for f in ir.prop_coll:
        rd["prop_coll"].append({
            "calculate_func": f.calculate_func,
            "prop_name": f.prop_name,
            "amp": f.amp,
            "Sbc": f.Sbc,
            "prop": f.prop,
            "all_paras": list(f.all_paras),
            "compl_paras": list(f.compl_paras),
            "theta": f.theta,
            "const": f.const,
            "damp": f.damp,
            "num_mod": f.num_mod,
            "const_index": list(f.const_index),
            "mod_name_list": [dict(d) for d in f.mod_name_list],
        })

    # args_collection (flattened from mods_collection)
    for cf, mods in ir.mods_collection.items():
        for mod_name, mod_data in mods.items():
            for k, v in mod_data.get("args", {}).items():
                rd["args_collection"][k] = v

    # mod_info (list of all mods, not grouped)
    for cf, mods in ir.mods_collection.items():
        for mod_name, mod_data in mods.items():
            rd["mod_info"].append(mod_data)

    # lh_coll
    for lh in ir.lh_coll:
        rd["lh_coll"].append({
            "tag": lh.tag,
            "slit_args_dict": lh.slit_args_dict,
            "trans_args_dict": {k: list(v) for k, v in lh.trans_args_dict.items()},
            "func_differ": [],
            "data_return_dict": lh.data_return_dict,
            "lasso_data_return_dict": lh.lasso_data_return_dict,
            "mc_return_dict": lh.mc_return_dict,
            "weight_return_dict": lh.weight_return_dict,
            "wt_data_return_dict": lh.wt_data_return_dict,
            "bounding": lh.bounding,
            "calc_wt": list(lh.calc_wt),
        "data_size": lh.data_size,
        "mc_size": lh.mc_size,
            "data_size": lh.data_size,
            "mc_size": lh.mc_size,
        })
        for f in lh.func_differ:
            rd["lh_coll"][-1]["func_differ"].append({
                "calculate_func": f.calculate_func,
                "prop_name": f.prop_name,
                "amp": f.amp,
                "Sbc": f.Sbc,
                "prop": f.prop,
                "all_paras": list(f.all_paras),
                "compl_paras": list(f.compl_paras),
                "theta": f.theta,
                "const": f.const,
                "damp": f.damp,
                "num_mod": f.num_mod,
                "const_index": list(f.const_index),
            })

    # range_dict (for boundary handling)
    for r in ir.ranges:
        rd["range_dict"][str(r.index)] = [r.lo, r.hi]

    return rd
