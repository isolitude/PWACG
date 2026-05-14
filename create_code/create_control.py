#!/usr/bin/env python3
# coding: utf-8
import json
import os
import re
import sys
import glob
import logging
from create_code import prepare_all_collection

log = logging.getLogger(__name__)


def _build_ir_from_legacy(ctrl):
    """Build PWAIR from the data already loaded by Prepare_All.

    Used during S2-S4 migration. S5 removed jinja2; this conversion
    happens upstream in create_all_scripts.py.
    """
    from create_code.schema.pwa_models import (
        PWAInfo, ModInfo, PropGroup, PropSpec, SbcSpec, ArgSpec,
    )
    from create_code.schema.generator_models import GeneratorConfig
    from create_code.schema.params_models import (
        ParametersFile, RunConfig, DataConfig, DrawConfig, DrawSwitch,
        PullOption, WeightOption, ModuleParams, CacheTensorEntry,
    )
    from create_code.ir.builder import build_ir

    # Build PWAInfo from all_mod_info (list of lists of raw dicts)
    all_mods: list = []
    for mod_group in ctrl.all_mod_info:
        for m in mod_group:
            # Build ArgSpec for each arg
            args: dict = {}
            for k, a in m.get("args", {}).items():
                rng = tuple(a["range"]) if "range" in a and a["range"] else None
                binding = a.get("binding")
                args[k] = ArgSpec(
                    value=a["value"],
                    name=a["name"],
                    fix=a.get("fix", False),
                    error=a.get("error", 0.0),
                    range=rng,
                    binding=binding,
                )
            # Build PropGroup
            propl = m["prop"]
            prop_phi = PropSpec(
                name=propl["prop_phi"]["name"],
                paras=tuple(propl["prop_phi"]["paras"]),
            )
            prop_f = PropSpec(
                name=propl["prop_f"]["name"],
                paras=tuple(propl["prop_f"]["paras"]),
            )
            prop = PropGroup(prop_phi=prop_phi, prop_f=prop_f)
            sbc = SbcSpec(phi=m["Sbc"]["phi"], f=m["Sbc"]["f"])
            all_mods.append(ModInfo(
                mod=m["mod"],
                amp=m["amp"],
                prop=prop,
                Sbc=sbc,
                args=args,
            ))

    pwa_info = PWAInfo(
        mod_info=tuple(all_mods),
        external_binding=getattr(ctrl, "_binding_point", {}) or {},
    )

    # Build GeneratorConfig from the original dict_generator
    gen_dict = {
        "id": ctrl.generator_id,
        "jinja_fit_info": ctrl.jinja_fit_info,
        "jinja_draw_info": ctrl.jinja_draw_info,
        "json_pwa": ctrl.json_pwa,
        "annex_info": ctrl.info,
    }
    gen_config = GeneratorConfig(**gen_dict)

    # Build ParametersFile
    # parameters dict: {"base": {"run_config": {...}, "data_config": {...}}, "fit": {...}, ...}
    module_params: dict = {}
    for mod_key, mod_val in ctrl.parameters.items():
        rc = mod_val["run_config"]
        dc = mod_val.get("data_config", {})
        module_params[mod_key] = ModuleParams(
            run_config=RunConfig(
                total_gpu_id=tuple(rc.get("total_gpu_id", ())),
                processes_gpus=rc.get("processes_gpus"),
                max_processes=rc.get("max_processes"),
                max_processes_memory=rc.get("max_processes_memory"),
                thread_gpus=rc.get("thread_gpus"),
                threads_in_one_gpu=rc.get("threads_in_one_gpu"),
            ),
            data_config=DataConfig(
                data_slices=dc.get("data_slices"),
                mc_slices=dc.get("mc_slices"),
                mini_run=dc.get("mini_run"),
            ),
        )

    draw_cfg = ctrl.draw_config
    params = ParametersFile(
        parameters=module_params,
        draw_config=DrawConfig(
            switch=DrawSwitch(
                likelihood=draw_cfg["switch"]["likelihood"],
                weight=draw_cfg["switch"]["weight"],
                pull=draw_cfg["switch"]["pull"],
                mods=draw_cfg["switch"]["mods"],
            ),
            pull_option=PullOption(
                bin=draw_cfg["pull_option"]["bin"],
                min=draw_cfg["pull_option"]["min"],
                max=draw_cfg["pull_option"]["max"],
            ),
            weight_option=WeightOption(
                bin=draw_cfg["weight_option"]["bin"],
                mods_num=draw_cfg["weight_option"]["mods_num"],
            ),
        ),
        CacheTensor={
            k: CacheTensorEntry(data=v["data"], mc=v["mc"])
            for k, v in ctrl.CacheTensor.items()
        },
    )

    from pathlib import Path
    return build_ir(pwa_info, gen_config, params, data_dir=Path(ctrl._data_dir),
                    external_binding=getattr(ctrl, "_binding_point", None))


class Create_Code(prepare_all_collection.Prepare_All):
    def __init__(self, dict_generator):
        super().__init__(dict_generator)
        self._data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    def read_pwa(self, key):
        self.all_mod_info = list()
        for addr_pwa_info in self.json_pwa[key]:
            filename = glob.glob(addr_pwa_info+"*")
            if filename:
                with open(filename[0], encoding='utf-8') as f:
                    dict_json = json.loads(f.read())
                    if "mod_info" in dict_json:
                        self.all_mod_info.append(dict_json["mod_info"])
                    if "external_binding" in dict_json:
                        self._binding_point = {**self._binding_point, **dict_json["external_binding"]}
            else:
                print(" Warning! No such file \"{}\", You should run fit create such file".format(addr_pwa_info))

    def _get_llm(self):
        """Lazy-init LLM client. Returns None if DEEPSEEK_API_KEY is not set."""
        try:
            from create_code.codegen.llm_client import LLMClient
            return LLMClient()
        except Exception as e:
            log.warning(f"LLM client unavailable: {e}")
            return None

    def _generate_run_script(self, ir, artifact_name, module, run_config, data_config, output_path):
        """Generate a run script via LLM codegen. Returns file content or None on failure."""
        try:
            from create_code.codegen.generator import CodeGenerator
            llm = self._get_llm()
            if llm is None:
                log.warning(f"[{artifact_name}] No LLM available")
                return None

            gen = CodeGenerator(llm=llm)
            extra = {
                "module": module,
                "run_config": run_config,
                "data_config": data_config,
            }
            content = gen.generate(ir, artifact_name, extra_context=extra)
            if content:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  [LLM] {artifact_name} -> {output_path}")
                return content
        except Exception as e:
            log.warning(f"[{artifact_name}] LLM generation failed: {e}")
        return None

    def _generate_code_script(self, ir, artifact_name, output_path, extra_context=None):
        """Generate a code script (CodeTemplate) via LLM codegen. Returns file content or None on failure."""
        try:
            from create_code.codegen.generator import CodeGenerator
            llm = self._get_llm()
            if llm is None:
                log.warning(f"[{artifact_name}] No LLM available")
                return None

            gen = CodeGenerator(llm=llm)
            content = gen.generate(ir, artifact_name, extra_context=extra_context)
            if content:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  [LLM] {artifact_name} -> {output_path}")
                return content
        except Exception as e:
            log.warning(f"[{artifact_name}] LLM generation failed: {e}")
        return None

    def generate_fit(self):
        print("generate_fit:")
        self.mod_info = sum(self.all_mod_info, [])
        self.prepare_all()

        # Build IR once for LLM codegen (shared across all fit modules)
        ir = None
        llm = self._get_llm()
        if llm is not None:
            try:
                ir = _build_ir_from_legacy(self)
            except Exception as e:
                log.warning(f"IR build failed: {e}")

        for module in self.jinja_fit_info.keys():
            merged_run = {**self.parameters["base"]["run_config"],
                          **self.parameters.get(module, {}).get("run_config", {})}
            merged_data = {**self.parameters["base"]["data_config"],
                           **self.parameters.get(module, {}).get("data_config", {})}
            self.render_dict.update(run_config=merged_run)
            self.render_dict.update(data_config=merged_data)
            address = self.jinja_fit_info[module]

            # CodeTemplate: use LLM
            code_path = "rendered_scripts/" + address["CodeScript"]
            if ir is not None and llm is not None:
                self._generate_code_script(
                    ir, module, code_path,
                    extra_context={"module": module, "run_config": merged_run, "data_config": merged_data}
                )
            # RunTemplate: use LLM
            run_path = "run/" + address["RunScript"]
            artifact_name = f"{module}_run"
            if ir is not None and llm is not None:
                self._generate_run_script(
                    ir, artifact_name, module, merged_run, merged_data, run_path
                )

    def generate_draw(self):
        print("generate_draw:")
        for n, mod_info in enumerate(self.all_mod_info):
            self.mod_info = mod_info
            self.prepare_all()

            # Build IR once per mod_info group
            ir = None
            llm = self._get_llm()
            if llm is not None:
                try:
                    ir = _build_ir_from_legacy(self)
                except Exception as e:
                    log.warning(f"IR build failed: {e}")

            for module in self.jinja_draw_info.keys():
                merged_run = {**self.parameters["base"]["run_config"],
                              **self.parameters.get(module, {}).get("run_config", {})}
                merged_data = {**self.parameters["base"]["data_config"],
                               **self.parameters.get(module, {}).get("data_config", {})}
                self.render_dict.update(run_config=merged_run)
                self.render_dict.update(data_config=merged_data)
                temp = self.render_dict
                address = self.jinja_draw_info[module]
                if module == "dplot" or module == "select":
                    with open("config/latex.json", encoding='UTF-8') as f:
                        latexjson = json.loads(f.read())
                        self.render_dict["sbc_collection"] = [sbc for sbc in list(latexjson["Sbc"].keys()) if re.match(".*"+self.render_dict["lh_coll"][0]["tag"],sbc)]
                if "ResultFile" in address:
                    self.render_dict.update(draw_result_file=address["ResultFile"][n])
                if "LassoResultFile" in address:
                    self.render_dict.update(lasso_result_file=address["LassoResultFile"][n])

                # CodeTemplate: use LLM
                code_path = "rendered_scripts/" + address["CodeScript"][n]
                if ir is not None and llm is not None:
                    self._generate_code_script(
                        ir, module, code_path,
                        extra_context={"module": module, "run_config": merged_run, "data_config": merged_data}
                    )
                # RunTemplate: use LLM
                run_path = "run/" + address["RunScript"]
                artifact_name = f"{module}_run"
                if ir is not None and llm is not None:
                    self._generate_run_script(
                        ir, artifact_name, module, merged_run, merged_data, run_path
                    )

                self.render_dict = temp

    def generate_tensor(self):
        print("generate_tensor:")
        self.initial_prepare()
        ir = None
        llm = self._get_llm()
        if llm is not None:
            try:
                ir = _build_ir_from_legacy(self)
            except Exception as e:
                log.warning(f"IR build failed: {e}")
        if ir is not None and llm is not None:
            self._generate_code_script(ir, "tensor", "run/RunCacheTensor.py")
