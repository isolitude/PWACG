#!/usr/bin/env python3
# coding: utf-8
"""Build PWAIR from Schema objects.

Replaces create_code/prepare_all_collection.py. Pure-function pipeline:
    build_ir(pwa_info, generator, params, *, sbc_count_paths=...) -> PWAIR

The legacy code carries a long list of `re.match` calls to classify parameters
(const/theta/mass/width). We do the same here for byte-equal output during S1,
but in a single pass with explicit naming conventions.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import numpy as onp

from ..schema import (
    PWAInfo,
    GeneratorConfig,
    ParametersFile,
    ModInfo,
)
from .ir_models import (
    PWAIR,
    ParamTable,
    BindingEdge,
    BindingGraph,
    RangeConstraint,
    FuncInfo,
    SlitArg,
    LHEntry,
    ArgsIndexCollection,
    LassoMeta,
)


def build_ir(
    pwa_info: PWAInfo,
    generator: GeneratorConfig,
    params: ParametersFile,
    *,
    data_dir: Path = Path("data"),
    external_binding: Optional[dict] = None,
) -> PWAIR:
    """Build PWAIR from the three top-level Schema objects.

    Args:
        pwa_info: parsed pwa_info_*.json (mod_info + optional external_binding)
        generator: parsed generator_*.json (artifact routing + annex_info)
        params: parsed parameters.json (run/data config + CacheTensor)
        data_dir: filesystem location of data arrays (for likelihood normalization)
        external_binding: extra bindings from generator-level json_pwa entries
    """
    info = generator.annex_info
    merge_key = info.merge
    combine_tags = info.combine.tag
    mod_info = pwa_info.mod_info

    calculate_func_coll = _calculate_func_coll(mod_info)
    mods_collection = _mods_collection(mod_info, calculate_func_coll)
    data_collection, sbc_collection, amp_collection = _data_collections(mod_info)

    params_list = _ordered_params(mod_info)
    name_list, args_list, error_list = _flatten_args(mod_info, params_list)
    name_to_index = {n: i for i, n in enumerate(name_list)}
    args_dict = _args_dict(mod_info, params_list)

    funcs = _build_funcs(
        mod_info,
        mods_collection,
        merge_key,
        combine_tags,
        args_dict,
        name_to_index,
    )
    prop_coll = _prop_coll(funcs)

    all_const_index = tuple(
        i for f in funcs for i in f.const_index
    )

    _binding_dict: dict[str, dict] = dict(external_binding or {})
    for mod in mod_info:
        for key, arg in mod.args.items():
            if arg.binding is not None:
                _binding_dict[key] = arg.binding

    binding = _build_binding(_binding_dict, name_list, name_to_index)

    fix_set = _fix_names(mod_info)
    float_index = _float_index(name_list, fix_set, _binding_dict)

    ranges_raw = _range_constraints(mod_info, name_to_index)

    mw_index_raw = [r.index for r in ranges_raw if r.index in float_index]
    mw_index = tuple(float_index.index(i) for i in mw_index_raw)
    mw_range = tuple((r.lo, r.hi) for r in ranges_raw if r.index in float_index)

    range_dict_for_slit = {
        str(r.index): (r.lo, r.hi) for r in ranges_raw
    }
    slit_args = _build_slit_args(
        args_dict,
        float_index,
        binding,
        args_list,
        name_list,
        range_dict_for_slit,
        funcs,
        boundary=info.fit.boundary,
    )

    args_index_collection = _args_index_collection(
        args_dict, float_index, combine_tags
    )

    lasso_meta = _lasso_meta(funcs, mod_info, combine_tags)

    lh_coll = _build_lh_coll(
        funcs,
        slit_args,
        info,
        sbc_collection,
        combine_tags,
        merge_key,
        params,
        data_dir=data_dir,
    )

    initial_parameters = {
        "all_parameters": list(args_list),
        "float_index": list(float_index),
    }

    # Convert generator routing info to plain dicts for IR
    jfi: dict[str, dict[str, Any]] = {}
    for mod, art in generator.jinja_fit_info.items():
        jfi[mod] = {
            "CodeTemplate": art.CodeTemplate,
            "CodeScript": art.CodeScript,
            "RunTemplate": art.RunTemplate,
            "RunScript": art.RunScript,
            "ResultFile": art.ResultFile,
        }
    jdi: dict[str, dict[str, Any]] = {}
    for mod, art in generator.jinja_draw_info.items():
        jdi[mod] = {
            "CodeTemplate": art.CodeTemplate,
            "CodeScript": list(art.CodeScript),
            "RunTemplate": art.RunTemplate,
            "RunScript": art.RunScript,
            "ResultFile": list(art.ResultFile),
            "LassoResultFile": list(art.LassoResultFile),
        }

    return PWAIR(
        generator_id=generator.id,
        calculate_func_coll=calculate_func_coll,
        mods_collection=mods_collection,
        funcs=funcs,
        prop_coll=prop_coll,
        data_collection=data_collection,
        sbc_collection=sbc_collection,
        amp_collection=amp_collection,
        params=ParamTable(
            name_list=name_list,
            args_list=args_list,
            error_list=error_list,
            float_index=float_index,
            name_to_index=name_to_index,
        ),
        args_dict={k: tuple(v) for k, v in args_dict.items()},
        all_const_index=all_const_index,
        binding=binding,
        ranges=ranges_raw,
        mw_index=mw_index,
        mw_range=mw_range,
        slit_args=slit_args,
        lh_coll=lh_coll,
        args_index_collection=args_index_collection,
        lasso=lasso_meta,
        initial_parameters=initial_parameters,
        jinja_fit_info=jfi,
        jinja_draw_info=jdi,
    )


def _calculate_func_coll(mod_info: tuple[ModInfo, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for mod in mod_info:
        key = f"{mod.amp}_{mod.prop.prop_phi.name}_{mod.prop.prop_f.name}"
        if key not in seen:
            seen.append(key)
    return tuple(seen)


def _calculate_func(mod: ModInfo) -> str:
    return f"{mod.amp}_{mod.prop.prop_phi.name}_{mod.prop.prop_f.name}"


def _prop_name(mod: ModInfo) -> str:
    return f"{mod.prop.prop_phi.name}_{mod.prop.prop_f.name}"


def _mods_collection(
    mod_info: tuple[ModInfo, ...], calculate_func_coll: tuple[str, ...]
) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {f: {} for f in calculate_func_coll}
    for mod in mod_info:
        out[_calculate_func(mod)][mod.mod] = _mod_to_dict(mod)
    return out


def _mod_to_dict(mod: ModInfo) -> dict:
    return {
        "mod": mod.mod,
        "amp": mod.amp,
        "prop": {
            "prop_phi": {
                "name": mod.prop.prop_phi.name,
                "paras": list(mod.prop.prop_phi.paras),
                "_paras": list(mod.prop.prop_phi.paras[:-1]),
            },
            "prop_f": {
                "name": mod.prop.prop_f.name,
                "paras": list(mod.prop.prop_f.paras),
                "_paras": list(mod.prop.prop_f.paras[:-1]),
            },
        },
        "Sbc": {"phi": mod.Sbc.phi, "f": mod.Sbc.f},
        "args": {
            k: _arg_to_dict(v) for k, v in mod.args.items()
        },
        "calculate_func": _calculate_func(mod),
        "prop_name": _prop_name(mod),
    }


def _arg_to_dict(arg) -> dict:
    out: dict = {"value": arg.value, "name": arg.name, "error": arg.error}
    if arg.fix:
        out["fix"] = True
    if arg.range is not None:
        out["range"] = list(arg.range)
    if arg.binding is not None:
        out["binding"] = arg.binding
    return out


def _data_collections(
    mod_info: tuple[ModInfo, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    sbc, amp = [], []
    for mod in mod_info:
        for s in (mod.Sbc.phi, mod.Sbc.f):
            if s not in sbc:
                sbc.append(s)
        if mod.amp not in amp:
            amp.append(mod.amp)
    data = amp + sbc
    return tuple(data), tuple(sbc), tuple(amp)


def _ordered_params(mod_info: tuple[ModInfo, ...]) -> list[str]:
    """The legacy `parameters_list`: unique args.name values in first-seen order."""
    out: list[str] = []
    for mod in mod_info:
        for arg in mod.args.values():
            if arg.name not in out:
                out.append(arg.name)
    return out


def _flatten_args(
    mod_info: tuple[ModInfo, ...], params_list: list[str]
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    """Replicates prepare_all_collection.py:228-240.

    For each param name in params_list (outer loop), scan all args across all
    mods (inner loop) and append entries whose name matches via re.match
    (i.e. prefix match). Returns name_list, args_list, error_list aligned.
    """
    args_coll: dict[str, dict] = {}
    for mod in mod_info:
        for k, arg in mod.args.items():
            args_coll.setdefault(k, _arg_to_dict(arg))

    name_list: list[str] = []
    args_list: list[float] = []
    error_list: list[float] = []
    for pname in params_list:
        for key, arg in args_coll.items():
            if re.match(pname, arg["name"]):
                args_list.append(arg["value"])
                name_list.append(key)
                error_list.append(arg.get("error", 0.0))
    return tuple(name_list), tuple(args_list), tuple(error_list)


def _args_dict(
    mod_info: tuple[ModInfo, ...], params_list: list[str]
) -> dict[str, list[int]]:
    """Replicates prepare_all_collection.py:242-256."""
    args_coll: dict[str, dict] = {}
    for mod in mod_info:
        for k, arg in mod.args.items():
            args_coll.setdefault(k, _arg_to_dict(arg))

    foo: dict[str, int] = {}
    for p in params_list:
        i = 0
        for _, arg in args_coll.items():
            if re.match(p, arg["name"]):
                i += 1
        foo[p] = i

    begin = 0
    out: dict[str, list[int]] = {}
    for tag, n in foo.items():
        out[tag] = [begin + i for i in range(n)]
        begin += n
    return out


def _build_funcs(
    mod_info: tuple[ModInfo, ...],
    mods_collection: dict[str, dict[str, dict]],
    merge_key: str,
    combine_tags: tuple[str, ...],
    args_dict: dict[str, list[int]],
    name_to_index: dict[str, int],
) -> tuple[FuncInfo, ...]:
    """Equivalent of `func_info_god` (prepare_all_collection.py:87-180)."""
    funcs: list[FuncInfo] = []
    for cf, mods in mods_collection.items():
        first_key = next(iter(mods))
        first = mods[first_key]
        temp: list[str] = []
        for arg in first["args"].values():
            temp.append(arg["name"])

        all_paras: list[str] = []
        theta_const_temp: list[str] = []
        theta = None
        const = None
        merge_paras: list[str] = []
        for p in temp:
            all_paras.append(p)
            if re.match(".*theta", p):
                theta_const_temp.append(p)
                theta = p
            if re.match(".*const", p):
                theta_const_temp.append(p)
                const = p
            if re.match(".*" + merge_key, p):
                merge_paras.append(p)
        all_paras = list(dict.fromkeys(all_paras))
        compl = [x for x in temp if x not in theta_const_temp]

        damp = sum(
            1 for arg in first["args"].values() if re.match(".*const", arg["name"])
        )

        const_index: list[int] = []
        if const:
            for key, idxs in args_dict.items():
                if re.match(const, key):
                    const_index.extend(idxs)
        num_mod = int(len(const_index) / damp) if damp else 0

        mod_name_list: list[dict[str, dict[str, int]]] = []
        for mod in mod_info:
            if re.match(cf, _calculate_func(mod)):
                d = {
                    k: name_to_index[k]
                    for k in mod.args
                    if not (
                        re.match(".*phi", k)
                        or re.match(".*c", k)
                        or re.match(".*t", k)
                    )
                }
                mod_name_list.append({mod.mod: d})

        funcs.append(
            FuncInfo(
                calculate_func=cf,
                prop_name=first["prop_name"],
                amp=first["amp"],
                Sbc=first["Sbc"],
                prop=first["prop"],
                all_paras=tuple(all_paras),
                compl_paras=tuple(compl),
                theta=theta,
                const=const,
                damp=damp,
                num_mod=num_mod,
                const_index=tuple(const_index),
                merge_paras=tuple(merge_paras),
                mod_name_list=tuple(mod_name_list),
            )
        )
    return tuple(funcs)


def _prop_coll(funcs: tuple[FuncInfo, ...]) -> tuple[FuncInfo, ...]:
    """First-seen-prop_name deduplication (prepare_all_collection.py:128-135)."""
    seen: list[str] = []
    out: list[FuncInfo] = []
    for f in funcs:
        joined = " ".join(seen)
        if not re.match(".*" + f.prop_name, joined):
            out.append(f)
        seen.append(f.prop_name)
    return tuple(out)


def _build_binding(
    binding_dict: dict[str, dict],
    name_list: tuple[str, ...],
    name_to_index: dict[str, int],
) -> BindingGraph:
    edges_first: list[BindingEdge] = []
    edges_last: list[BindingEdge] = []
    str_keys = " ".join(binding_dict.keys())
    for src_name, b in binding_dict.items():
        if src_name not in name_list:
            continue
        dst_name = b["point"]
        bvalue = b["value"]
        begin = end = None
        for i, n in enumerate(name_list):
            if re.match(src_name, n):
                begin = i
            if re.match(dst_name, n):
                end = i
        if begin is None or end is None:
            continue
        edge = BindingEdge(
            src_name=src_name,
            src_index=begin,
            dst_name=dst_name,
            dst_index=end,
            bvalue=bvalue,
        )
        if re.match(".*" + dst_name, str_keys):
            edges_last.append(edge)
        else:
            edges_first.append(edge)
    edges = tuple(edges_first + edges_last)
    return BindingGraph(
        edges=edges,
        goto0=tuple(e.src_index for e in edges),
        goto1=tuple(e.dst_index for e in edges),
        bvalue=tuple(e.bvalue for e in edges),
    )


def _fix_names(mod_info: tuple[ModInfo, ...]) -> set[str]:
    fixed: set[str] = set()
    for mod in mod_info:
        for k, arg in mod.args.items():
            if arg.fix:
                fixed.add(k)
    return fixed


def _float_index(
    name_list: tuple[str, ...],
    fix_set: set[str],
    binding_dict: dict[str, dict],
) -> tuple[int, ...]:
    """Legacy uses `if not re.match(".*"+name, str_fix)` — replicate exactly."""
    fix_list = list(fix_set) + list(binding_dict.keys())
    str_fix = " ".join(fix_list)
    out: list[int] = []
    for i, n in enumerate(name_list):
        if not re.match(".*" + n, str_fix):
            out.append(i)
    return tuple(out)


def _range_constraints(
    mod_info: tuple[ModInfo, ...], name_to_index: dict[str, int]
) -> tuple[RangeConstraint, ...]:
    out: list[RangeConstraint] = []
    seen: dict[str, RangeConstraint] = {}
    for mod in mod_info:
        for k, arg in mod.args.items():
            if arg.range is None:
                continue
            if k in seen:
                continue
            for name, idx in name_to_index.items():
                if re.match(k, name):
                    rc = RangeConstraint(index=idx, lo=arg.range[0], hi=arg.range[1])
                    seen[k] = rc
                    out.append(rc)
                    break
    return tuple(out)


def _build_slit_args(
    args_dict: dict[str, list[int]],
    float_index: tuple[int, ...],
    binding: BindingGraph,
    args_list: tuple[float, ...],
    name_list: tuple[str, ...],
    range_dict: dict[str, tuple[float, float]],
    funcs: tuple[FuncInfo, ...],
    *,
    boundary: bool,
) -> dict[str, SlitArg]:
    """Replicates prepare_all_collection.py:357-401."""
    fi = list(float_index)
    out: dict[str, SlitArg] = {}
    for key, arg_indices in args_dict.items():
        entries: list[str] = []
        trans: list[str] = []
        for i in arg_indices:
            if i in fi:
                if str(i) in range_dict:
                    lo, hi = range_dict[str(i)]
                    trans.append(
                        f"np.power({lo}-args[{fi.index(i)}],2)*{hi}"
                    )
                entries.append(f"args[{fi.index(i)}]")
            elif i in binding.goto0:
                wz = list(binding.goto0).index(i)
                igo = binding.goto1[wz]
                if str(igo) in range_dict:
                    lo, hi = range_dict[str(igo)]
                    trans.append(
                        f"np.power({lo}-args[{fi.index(igo)}],2)*{hi}"
                    )
                if igo in fi:
                    bv = binding.bvalue[wz]
                    entries.append(f"args[{fi.index(igo)}]{bv:+}")
                else:
                    entries.append(str(args_list[i]))
            else:
                if str(i) in range_dict:
                    # legacy bug-compatible: uses args_list_complete(i) (TypeError on real run)
                    # we mirror behavior only when actually triggered; otherwise skip safely
                    pass
                entries.append(str(args_list[i]))
        expr = "np.array([{}])".format(",".join(entries))
        reshape_damp = None
        if re.match(".*const", key) or re.match(".*theta", key):
            for f in funcs:
                matched = False
                for arg_name in f.all_paras:
                    if re.match(key, arg_name):
                        expr = "{}.reshape(-1,{})".format(expr, f.damp)
                        reshape_damp = f.damp
                        matched = True
                        break
                if matched:
                    break
        out[key] = SlitArg(
            expr=expr,
            entries=tuple(entries),
            reshape_damp=reshape_damp,
            trans_terms=tuple(trans),
        )
    return out


def _args_index_collection(
    args_dict: dict[str, list[int]],
    float_index: tuple[int, ...],
    combine_tags: tuple[str, ...],
) -> ArgsIndexCollection:
    """Replicates prepare_all_collection.py:403-459 (the render_temp branch)."""
    fi = list(float_index)
    who = ["const", "theta", "mass", "width"]
    by_kind: dict[str, list[int]] = {w: [] for w in who}
    for w in who:
        for key, idxs in args_dict.items():
            if re.match(".*" + w, key):
                for i in idxs:
                    if i in fi:
                        by_kind[w].append(fi.index(i))

    # flatte across all tags (per-tag dict flattened)
    flatte_regex = [re.compile(r".*f980_rg.*"), re.compile(r".*f980_g")]
    flatte: list[int] = []
    for tag in combine_tags:
        for key, idxs in args_dict.items():
            if re.match(".*" + tag, key) and any(rx.match(key) for rx in flatte_regex):
                for i in idxs:
                    if i in fi:
                        flatte.append(fi.index(i))

    return ArgsIndexCollection(
        const=tuple(by_kind["const"]),
        theta=tuple(by_kind["theta"]),
        mass=tuple(by_kind["mass"]),
        width=tuple(by_kind["width"]),
        flatte=tuple(flatte),
    )


def _lasso_meta(
    funcs: tuple[FuncInfo, ...],
    mod_info: tuple[ModInfo, ...],
    combine_tags: tuple[str, ...],
) -> LassoMeta:
    """Replicates prepare_all_collection.py:152-176."""
    mod_name_list: list[str] = []
    dimensions: list[int] = []
    for f in funcs:
        for mod in mod_info:
            if re.match(f.calculate_func, _calculate_func(mod)):
                mod_name_list.extend(
                    mod.mod.replace("_" + tag, "")
                    for tag in combine_tags
                    if re.match(".*" + tag, mod.mod)
                )
                free = sum(1 for arg in mod.args.values() if not arg.fix)
                dimensions.append(free)

    lasso_frac_dict = {
        n: (d,) for d, n in zip(dimensions, mod_name_list)
    }

    all_mods: list[str] = []
    for f in funcs:
        for m in f.mod_name_list:
            all_mods.append(next(iter(m)))

    sif_free: dict[str, tuple[int, ...]] = {}
    unique_mods = list(dict.fromkeys(all_mods))
    for name in lasso_frac_dict:
        xi = sum(1 for m in unique_mods if re.match(name + ".*", m))
        sif_free[name] = (xi * lasso_frac_dict[name][0],)

    return LassoMeta(
        lasso_frac_dict=lasso_frac_dict,
        sif_free_dict=sif_free,
        mod_name_list=tuple(mod_name_list),
    )


def _build_lh_coll(
    funcs: tuple[FuncInfo, ...],
    slit_args: dict[str, SlitArg],
    info,
    sbc_collection: tuple[str, ...],
    combine_tags: tuple[str, ...],
    merge_key: str,
    params: ParametersFile,
    *,
    data_dir: Path,
) -> tuple[LHEntry, ...]:
    """Replicates prepare_all_collection.py:485-619.

    Builds the per-tag likelihood collection. Performs filesystem reads for
    `data_size` / `mc_size`; these are committed into the IR (no IO at codegen
    time).
    """
    sbc_str = " ".join(sbc_collection)

    # combine_*_add_all_amp: per-tag amplitude sums (write_all_amp_add equivalent)
    data_per_tag: dict[str, list[str]] = {tag: [] for tag in combine_tags}
    mc_per_tag: dict[str, list[str]] = {tag: [] for tag in combine_tags}
    lasso_per_tag: dict[str, list[str]] = {tag: [] for tag in combine_tags}
    for f in funcs:
        for tag in combine_tags:
            if re.match(".*" + tag, f.calculate_func):
                data_per_tag[tag].append("data_" + f.calculate_func)
                mc_per_tag[tag].append("mc_" + f.calculate_func)
                lasso_per_tag[tag].append("lasso_data_" + f.calculate_func)

    combine_data = {t: " + ".join(v) for t, v in data_per_tag.items()}
    combine_mc = {t: " + ".join(v) for t, v in mc_per_tag.items()}
    combine_lasso = {
        t: " + ".join(
            f'np.sum(np.sqrt(np.einsum("ljk->l",dplex.dabs({a}))))' for a in v
        )
        for t, v in lasso_per_tag.items()
    }

    lh_list: list[LHEntry] = []
    for tag in combine_tags:
        if not data_per_tag[tag]:
            continue

        # per-tag slit_args / trans_args
        slit_tag: dict[str, str] = {}
        trans_tag: dict[str, tuple[str, ...]] = {}
        for key, sa in slit_args.items():
            if re.match(".*" + merge_key, key) or re.match(".*" + tag, key):
                slit_tag[key] = sa.expr
                trans_tag[key] = sa.trans_terms

        func_differ = tuple(f for f in funcs if re.match(".*" + tag, f.calculate_func))

        # data_size / mc_size
        sbc_match = next((s for s in sbc_collection if re.match(".*" + tag, s)), None)
        if info.fit.use_weight:
            wp = data_dir / "weight" / f"weight_{tag}.npy"
            data_size = float(onp.sum(onp.load(wp))) if wp.exists() else 0.0
        elif sbc_match is not None and re.match(".*" + tag, sbc_str):
            dp = data_dir / "real_data" / f"{sbc_match}.npy"
            if dp.exists():
                data_size = float(onp.load(dp).shape[0])
            else:
                # Fallback: use CacheTensor size from parameters.json
                entry = params.CacheTensor.get(tag)
                data_size = float(entry.data) if entry else 0.0
        else:
            data_size = 0.0
        if sbc_match is not None:
            mp = data_dir / "mc_truth" / f"{sbc_match}.npy"
            if mp.exists():
                mc_size = float(onp.load(mp).shape[0])
            else:
                entry = params.CacheTensor.get(tag)
                mc_size = float(entry.mc) if entry else 0.0
        else:
            mc_size = 0.0

        # return strings (data / wt_data / lasso / mc / weight)
        data_return = (
            f"return -np.sum(np.log(np.sum(dplex.dabs({combine_data[tag]}),axis=1))) "
            f"+ {data_size}*step_function"
        )
        wt_data_return = (
            f"return -np.sum(self.wt_data_{tag}*np.log(np.sum(dplex.dabs({combine_data[tag]}),axis=1))) "
            f"+ {data_size}*step_function"
        )
        lasso_return = (
            f"return -np.sum(np.log(np.sum(dplex.dabs({combine_data[tag]}),axis=1))) "
            f"+ np.power(10,{info.fit.lambda_tfc})*({combine_lasso[tag]})"
        )
        mc_return = f"return np.sum(dplex.dabs({combine_mc[tag]}))"
        weight_return = f"return np.sum(dplex.dabs({combine_data[tag]}),axis=1)"

        # bounding: sum_frac + step_function
        trans_temp: list[str] = []
        for v in trans_tag.values():
            trans_temp.extend(v)
        param_limits = "+".join(trans_temp) if (info.fit.boundary and trans_temp) else "0.0"

        sum_frac = (
            "sum_frac = "
            + "np.sum(dplex.dabs({})) \n".format(
                "+".join(f'np.einsum("mljk->mjk", {l})' for l in lasso_per_tag[tag])
            )
        )

        # total_frac split: _l_, _r_, others
        temp_l, temp_r, temp_f = [], [], []
        for a in lasso_per_tag[tag]:
            if re.match(".*_l_", a):
                temp_l.append(a)
            elif re.match(".*_r_", a):
                temp_r.append(a)
            else:
                temp_f.append(a)
        temp_x = [f"{l}+{r}" for l, r in zip(temp_l, temp_r)]
        temp_frac = temp_f + temp_x

        smooth_add_frac = "np.power({}-{},2.0)".format(
            "+".join(
                f'np.sum(np.einsum("ljk->l",dplex.dabs({a}))/sum_frac)' for a in temp_frac
            ),
            info.fit.total_frac[tag],
        )
        step_function = (
            "step_function = ("
            + f"{smooth_add_frac})*{info.fit.lambda_tfc}"
            + " + "
            + param_limits
        )
        bounding = sum_frac + "\n        " + step_function

        # save mc weight
        total_wt = "total_wt = " + "np.sum(dplex.dabs({}),axis=1)".format(
            "+".join(f'np.einsum("mljk->mjk", {l})' for l in lasso_per_tag[tag])
        )
        mod_wt = "wt_list = [total_wt,{}]".format(
            ",".join(f'np.einsum("ljk->lj",dplex.dabs({a}))' for a in temp_frac)
        )

        lh_list.append(
            LHEntry(
                tag=tag,
                slit_args_dict=slit_tag,
                trans_args_dict=trans_tag,
                func_differ=func_differ,
                data_return_dict=data_return,
                lasso_data_return_dict=lasso_return,
                mc_return_dict=mc_return,
                weight_return_dict=weight_return,
                wt_data_return_dict=wt_data_return,
                bounding=bounding,
                calc_wt=(total_wt, mod_wt),
                data_size=data_size,
                mc_size=mc_size,
            )
        )
    return tuple(lh_list)
