#!/usr/bin/env python3
# coding: utf-8
"""
LLM 版本的 create_control.py
使用 LLM Agent 替代 Jinja2 模板功能
"""

import json
import os
import glob
from typing import Dict, Any
from .code_generator import PWACodeGenerator
from create_code import prepare_all_collection


class LLMCreateCode(prepare_all_collection.Prepare_All):
    """使用 LLM 的代码创建类"""
    
    def __init__(self, dict_generator: Dict[str, Any], agent_config: Dict[str, Any]):
        super().__init__(dict_generator)
        self.agent_config = agent_config
        self.code_generator = PWACodeGenerator(agent_config)
    
    def read_pwa(self, key: str):
        """读取 PWA 配置数据（继承原有逻辑）"""
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
                print(f" Warning! No such file \"{addr_pwa_info}\", You should run fit create such file")
    
    def llm_generate_fit(self):
        """使用 LLM 生成拟合代码"""
        print("LLM 生成拟合代码:")
        self.mod_info = sum(self.all_mod_info, [])
        self.prepare_all()
        
        for module in self.jinja_fit_info.keys():
            # 更新渲染字典
            self.render_dict.update(
                run_config={**self.parameters["base"]["run_config"], 
                           **self.parameters[module]["run_config"]}
            )
            self.render_dict.update(
                data_config={**self.parameters["base"]["data_config"], 
                            **self.parameters[module]["data_config"]}
            )
            
            address = self.jinja_fit_info[module]
            
            # 准备模板数据
            template_data = {
                "module_name": module,
                "render_dict": self.render_dict,
                "result_file": address["ResultFile"],
                "template_type": "fit",
                "original_template": address["CodeTemplate"],
                "run_template": address["RunTemplate"]
            }
            
            # 生成代码脚本
            code_output_path = f"rendered_scripts/{address['CodeScript']}"
            success = self.code_generator.generate_and_save(
                template_data, code_output_path, "fit"
            )
            
            if success:
                print(f"✓ 生成代码脚本: {code_output_path}")
            else:
                print(f"✗ 生成代码脚本失败: {code_output_path}")
            
            # 生成运行脚本
            run_template_data = {**template_data, "template_type": "run_fit"}
            run_output_path = f"run/{address['RunScript']}"
            success = self.code_generator.generate_and_save(
                run_template_data, run_output_path, "run"
            )
            
            if success:
                print(f"✓ 生成运行脚本: {run_output_path}")
            else:
                print(f"✗ 生成运行脚本失败: {run_output_path}")
    
    def llm_generate_draw(self):
        """使用 LLM 生成绘图代码"""
        print("LLM 生成绘图代码:")
        
        for n, mod_info in enumerate(self.all_mod_info):
            self.mod_info = mod_info
            self.prepare_all()
            
            for module in self.jinja_draw_info.keys():
                # 更新渲染字典
                self.render_dict.update(
                    run_config={**self.parameters["base"]["run_config"], 
                               **self.parameters[module]["run_config"]}
                )
                self.render_dict.update(
                    data_config={**self.parameters["base"]["data_config"], 
                                **self.parameters[module]["data_config"]}
                )
                
                temp = self.render_dict.copy()
                address = self.jinja_draw_info[module]
                
                # 特殊处理某些模块
                if module == "dplot" or module == "select":
                    try:
                        with open("config/latex.json", encoding='UTF-8') as f:
                            latexjson = json.loads(f.read())
                            self.render_dict["sbc_collection"] = [
                                sbc for sbc in list(latexjson["Sbc"].keys()) 
                                if self.render_dict["lh_coll"][0]["tag"] in sbc
                            ]
                    except Exception as e:
                        print(f"Warning: 读取 latex.json 失败: {e}")
                
                # 添加结果文件信息
                if "ResultFile" in address:
                    self.render_dict.update(draw_result_file=address["ResultFile"][n])
                if "LassoResultFile" in address:
                    self.render_dict.update(lasso_result_file=address["LassoResultFile"][n])
                
                # 准备模板数据
                template_data = {
                    "module_name": module,
                    "render_dict": self.render_dict,
                    "template_type": "draw",
                    "original_template": address["CodeTemplate"],
                    "run_template": address["RunTemplate"],
                    "iteration": n
                }
                
                # 生成代码脚本
                code_output_path = f"rendered_scripts/{address['CodeScript'][n]}"
                success = self.code_generator.generate_and_save(
                    template_data, code_output_path, "draw"
                )
                
                if success:
                    print(f"✓ 生成绘图脚本: {code_output_path}")
                else:
                    print(f"✗ 生成绘图脚本失败: {code_output_path}")
                
                # 生成运行脚本
                run_template_data = {**template_data, "template_type": "run_draw"}
                run_output_path = f"run/{address['RunScript']}"
                success = self.code_generator.generate_and_save(
                    run_template_data, run_output_path, "run"
                )
                
                if success:
                    print(f"✓ 生成运行脚本: {run_output_path}")
                else:
                    print(f"✗ 生成运行脚本失败: {run_output_path}")
                
                self.render_dict = temp
    
    def llm_generate_tensor(self):
        """使用 LLM 生成张量计算代码"""
        print("LLM 生成张量计算代码:")
        self.initial_prepare()
        
        template_data = {
            "module_name": "tensor",
            "render_dict": self.render_dict,
            "template_type": "tensor",
            "original_template": "Tensor/RunCacheTensor.py"
        }
        
        output_path = "run/RunCacheTensor.py"
        success = self.code_generator.generate_and_save(
            template_data, output_path, "tensor"
        )
        
        if success:
            print(f"✓ 生成张量计算脚本: {output_path}")
        else:
            print(f"✗ 生成张量计算脚本失败: {output_path}")
    
    def generate_all(self):
        """生成所有类型的代码"""
        print("=== 开始使用 LLM 生成所有代码 ===")
        
        # 生成拟合代码
        self.initial_prepare()
        self.read_pwa("fit")
        self.llm_generate_fit()
        
        # 生成绘图代码
        self.initial_prepare()
        self.read_pwa("draw")
        self.llm_generate_draw()
        
        # 生成张量计算代码
        self.llm_generate_tensor()
        
        print("=== LLM 代码生成完成 ===")


def main():
    """主函数，用于测试"""
    # 检查目录结构
    def check_directory_structure():
        father_dirs = ["output", "result_repo", "rendered_scripts", "run"]
        child_dirs = [
            'output/fit/fit_result_combine', 'output/fit/fit_result_kk', 
            'output/fit/fit_result_pipi', 'output/error', 
            'output/pictures/partial_mods_pictures', 'output/draw', 
            'output/lasso', 'output/pull', 'output/select', 'output/significance'
        ]
        required_dirs = father_dirs + child_dirs
        
        for subdir in required_dirs:
            if not os.path.isdir(subdir):
                print(f"目录 {subdir} 不存在！创建新目录")
                os.makedirs(subdir, exist_ok=True)
        return True
    
    check_directory_structure()
    
    # Agent 配置
    agent_config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'openai_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        'model': 'gpt-3.5-turbo'
    }
    
    # 加载生成器配置
    try:
        with open("config/generator_kk.json", encoding='utf-8') as f:
            dict_json = json.loads(f.read())
            
        # 创建 LLM 代码生成器
        llm_creator = LLMCreateCode(dict_json, agent_config)
        llm_creator.generate_all()
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()