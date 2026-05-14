#!/usr/bin/env python3
# coding: utf-8
"""PWAIR — Intermediate Representation for partial wave analysis code generation.

This module defines structured, deterministic models that replace the regex+string
soup in prepare_all_collection.py. The builder (ir/builder.py) turns Schema objects
into a PWAIR; the codegen layer (codegen/) consumes PWAIR to produce Python code.

Determinism contract:
    Same input (PWAInfo + GeneratorConfig + ParametersFile + filesystem state)
    must produce byte-equal `model_dump_json(by_alias=True)` output.
"""
from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel


class ParamTable(BaseModel, frozen=True):
    """Flat view of all parameters across mod_info entries.

    Mirrors the legacy `name_list_complete` / `args_list_complete` / `error_list`
    that prepare_all_collection.py:204-241 builds via regex matching.
    """
    name_list: tuple[str, ...]
    args_list: tuple[float, ...]
    error_list: tuple[float, ...]
    float_index: tuple[int, ...]
    name_to_index: dict[str, int]


class BindingEdge(BaseModel, frozen=True):
    """A single binding constraint: source parameter follows target parameter."""
    src_name: str
    src_index: int
    dst_name: str
    dst_index: int
    bvalue: float


class BindingGraph(BaseModel, frozen=True):
    """Topologically sorted binding DAG.

    Replaces prepare_all_collection.py:273-311 dict-shuffling. Edges are ordered
    such that any edge whose dst_name itself appears as src in another edge comes
    last (consistent with the legacy `first` / `last` partitioning).
    """
    edges: tuple[BindingEdge, ...]
    goto0: tuple[int, ...]
    goto1: tuple[int, ...]
    bvalue: tuple[float, ...]


class RangeConstraint(BaseModel, frozen=True):
    """Soft boundary penalty: lo and hi from pwa_info range field."""
    index: int
    lo: float
    hi: float


class FuncInfo(BaseModel, frozen=True):
    """One amplitude/propagator combination (unique by calculate_func).

    Fields mirror the entries pushed into `func_info` by
    prepare_all_collection.py:87-180.
    """
    calculate_func: str
    prop_name: str
    amp: str
    Sbc: dict[str, str]
    prop: dict[str, dict[str, Any]]
    all_paras: tuple[str, ...]
    compl_paras: tuple[str, ...]
    theta: Optional[str] = None
    const: Optional[str] = None
    damp: int = 0
    num_mod: int = 0
    const_index: tuple[int, ...] = ()
    merge_paras: tuple[str, ...] = ()
    mod_name_list: tuple[dict[str, dict[str, int]], ...] = ()


class SlitArg(BaseModel, frozen=True):
    """Parameter-unpacking expression for one logical parameter group.

    Legacy form (prepare_all_collection.py:357-401) is a raw Python string like
    `np.array([args[3], args[5]]).reshape(-1, 5)`. We keep the string form
    here for byte-equal codegen during S1, but the codegen prompt can rebuild
    it from `entries` + `reshape_damp` in later stages.
    """
    expr: str
    entries: tuple[str, ...]
    reshape_damp: Optional[int] = None
    trans_terms: tuple[str, ...] = ()


class LHEntry(BaseModel, frozen=True):
    """One likelihood collection (per tag in combine.tag)."""
    tag: str
    slit_args_dict: dict[str, str]
    trans_args_dict: dict[str, tuple[str, ...]]
    func_differ: tuple[FuncInfo, ...]
    data_return_dict: str
    lasso_data_return_dict: str
    mc_return_dict: str
    weight_return_dict: str
    wt_data_return_dict: str
    bounding: str
    calc_wt: tuple[str, str]
    data_size: float
    mc_size: float


class ArgsIndexCollection(BaseModel, frozen=True):
    const: tuple[int, ...]
    theta: tuple[int, ...]
    mass: tuple[int, ...]
    width: tuple[int, ...]
    flatte: tuple[int, ...]


class LassoMeta(BaseModel, frozen=True):
    lasso_frac_dict: dict[str, tuple[int, ...]]
    sif_free_dict: dict[str, tuple[int, ...]]
    mod_name_list: tuple[str, ...]


class PWAIR(BaseModel, frozen=True):
    """Top-level intermediate representation.

    All fields are deterministic functions of (PWAInfo, GeneratorConfig,
    ParametersFile, file-system measurements like data array shapes).
    """
    generator_id: str
    calculate_func_coll: tuple[str, ...]
    mods_collection: dict[str, dict[str, dict[str, Any]]]
    funcs: tuple[FuncInfo, ...]
    prop_coll: tuple[FuncInfo, ...]
    data_collection: tuple[str, ...]
    sbc_collection: tuple[str, ...]
    amp_collection: tuple[str, ...]
    params: ParamTable
    args_dict: dict[str, tuple[int, ...]]
    all_const_index: tuple[int, ...]
    binding: BindingGraph
    ranges: tuple[RangeConstraint, ...]
    mw_index: tuple[int, ...]
    mw_range: tuple[tuple[float, float], ...]
    slit_args: dict[str, SlitArg]
    lh_coll: tuple[LHEntry, ...]
    args_index_collection: ArgsIndexCollection
    lasso: LassoMeta
    initial_parameters: dict[str, Any]
    jinja_fit_info: dict[str, dict[str, Any]] = {}
    jinja_draw_info: dict[str, dict[str, Any]] = {}
