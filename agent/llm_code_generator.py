#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import toml
import hashlib

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from agent.easytrans_client import EasyTransClient, EasyTransError


# Section mapping for programmatic access
TEMPLATE_SECTIONS = {
    'COMMON_UTILITIES': 'Common utility functions and imports',
    'PATH_CONFIG': 'Path configuration and setup', 
    'LOGGING_CONFIG': 'Logging configuration functions',
    'PHYSICS_FUNCTIONS': 'Physics calculation functions (resonance shapes)',
    'DATA_LOADING': 'Data loading function templates',
    'RESONANCE_CALCULATIONS': 'Composite resonance calculation templates'
}

def parse_template_sections(file_path=__file__):
    """
    Parse template sections for programmatic access.
    
    Returns:
        dict: Dictionary mapping section names to their content
    """
    sections = {}
    current_section = None
    section_content = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.strip().startswith('# SECTION:'):
            # Save previous section
            if current_section:
                sections[current_section] = ''.join(section_content)
            
            # Start new section
            current_section = line.strip().split('SECTION:')[1].strip()
            section_content = []
        elif current_section:
            section_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = ''.join(section_content)
    
    return sections

def checkDirectoryStructure():
    # 检查目录中是否存在指定的文件和子目录
    fatherDirs = ["output", "result_repo", "rendered_scripts", "run"]
    childDirs = ['output/fit/fit_result_combine', 'output/fit/fit_result_kk', 'output/fit/fit_result_pipi', 'output/error', 'output/pictures/partial_mods_pictures', 'output/draw', 'output/lasso', 'output/pull', 'output/select', 'output/significance', 'agent/cache']
    requiredDirs = fatherDirs + childDirs
    for subdir in requiredDirs:
        if not os.path.isdir(subdir):
            print(f"Directory {subdir} not exists！ create directory")
            os.system("mkdir -p {}".format(subdir))
    return True


class LLMResonanceGenerator:
    """LLM驱动的共振态代码生成器"""
    
    def __init__(self, config_path: str = "agent/resonances_config.toml", model: str = "o3-pro-2025-06-10", model_check: str = "gpt-5-2025-08-07"):
        """
        初始化LLM代码生成器
        
        Args:
            config_path: TOML配置文件路径
            model: 使用的LLM模型
        """
        # 检查并创建必要的目录结构
        checkDirectoryStructure()

        # 初始化EasyTrans客户端
        self.llm_client = EasyTransClient()
        print(f"🤖 LLM引擎初始化完成: {model}")
        
        self.config_path = config_path
        self.model = model
        self.model_check = model_check
        self.config = self.load_config()

        # 解析模板部分
        self.sections = parse_template_sections("agent/common_template.py")
        
        self.system_prompt = """You are a professional physics computation code generator, specializing in Partial Wave Analysis (PWA) in particle physics.Generate complete, runnable Python functions without overly detailed docstrings."""
    
    def _load_cache(self, cache_file) -> dict:
        """从文件中加载缓存。如果文件不存在或无效，则返回一个空字典。"""
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    print(f"✅ 成功从 {cache_file} 加载缓存。")
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 加载缓存文件 {cache_file} 失败: {e}。将创建一个新的缓存。")
        
        print("ℹ️ 未找到缓存文件或文件为空，将创建一个新的缓存。")
        return {}

    def _save_cache(self, load_data_cache, cache_file):
        """将当前内存中的缓存保存到文件。"""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(load_data_cache, f, ensure_ascii=False, indent=4)
            print(f"💾 缓存已成功保存到 {cache_file}。")
        except IOError as e:
            print(f"❌ 保存缓存到 {cache_file} 失败: {e}。")
        
    def load_config(self) -> Dict[str, Any]:
        """
        从TOML文件加载共振态配置
        
        Returns:
            Dict[str, Any]: 包含所有共振态配置的字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = toml.load(f)
            
            print(f"✅ 配置文件加载成功: {self.config_path}")
            print(f"📊 找到 {len(config_data.get('resonances', {}))} 个共振态配置")
            
            return config_data
            
        except FileNotFoundError:
            print(f"❌ 配置文件未找到: {self.config_path}")
            raise
        except toml.TomlDecodeError as e:
            print(f"❌ TOML文件解析失败: {e}")
            raise
        except Exception as e:
            print(f"❌ 配置加载失败: {e}")
            raise
    
    def extract_resonance_config(self, resonance_name: str) -> Dict[str, Any]:
        """
        提取指定共振态的配置信息
        
        Args:
            resonance_name: 共振态名称 (如 'phif0_980', 'phif2_1270')
            
        Returns:
            Dict[str, Any]: 共振态配置字典
        """
        resonances = self.config.get('resonances', {})
        
        if resonance_name not in resonances:
            available_resonances = list(resonances.keys())
            raise ValueError(f"未找到共振态 '{resonance_name}'。可用的共振态: {available_resonances}")
        
        resonance_config = resonances[resonance_name]
        
        # 构造返回的配置结构
        config = {
            'name': resonance_name,
            'config': resonance_config
        }
        
        return config
    
    def get_all_resonance_names(self) -> List[str]:
        """获取所有可用的共振态名称"""
        return list(self.config.get('resonances', {}).keys())
    
    def get_all_resonance_data(self) -> List[str]:
        # 提取所有 Sbc
        sbc_list = [
            prop["Sbc"]
            for resonance in self.config["resonances"].values()
            for prop in resonance.get("propagators", {}).values()
            if "Sbc" in prop
        ]
        # 提取所有 AMP
        amp_list = [
            resonance["Amplitude"]["AMP"]
            for resonance in self.config["resonances"].values()
            if "AMP" in resonance.get("Amplitude", {})
        ]

        return list(dict.fromkeys(sbc_list)), list(dict.fromkeys(amp_list))
    
    def print_config_summary(self):
        """打印配置文件摘要信息"""
        print("🔍 配置文件摘要:")
        print("=" * 40)
        
        resonances = self.config.get('resonances', {})
        for name, config in resonances.items():
            print(f"📋 共振态: {name}")
            
            # 打印传播子信息
            propagators = config.get('propagators', {})
            for prop_name, prop_config in propagators.items():
                prop_type = prop_config.get('propagator_type', 'unknown')
                print(f"   - {prop_name}: {prop_type}")
            
            # 打印系数数量
            Amplitude = config.get('Amplitude', {})
            const_count = len([k for k in Amplitude.keys() if k.startswith('const')])
            theta_count = len([k for k in Amplitude.keys() if k.startswith('theta')])
            print(f"   - 系数: {const_count} const, {theta_count} theta")
    
    def analysis_toml_config_prompt(self) -> Dict[str, Any]:
        print(f"🚀 开始分析共振态配置以进行分组...")
        all_resonances_info = self.config.get('resonances', {})
        all_resonances_info = json.dumps(all_resonances_info, indent=4)
        prompt = f"""
toml 配置：
{all_resonances_info}（注：此处为变量，指“所有共振信息”）

任务：
1. 按“传播子类型”（propagator_type）对共振（resonances）进行分组。
    定义：
    - 相似的“A传播子”（A_propagator）必须具有相同的“传播子类型”（propagator_type）。
    - 相似的“B传播子”（B_propagator）必须具有相同的“传播子类型”（propagator_type）。
    组名称：
    将“A传播子”（A_propagator）类型与“B传播子”（B_propagator）类型用下划线（_）组合。示例："TypeA_TypeB"（即“A类型_B类型”）。

2. 因为在计算的时候，同一类型的传播子可以向量化计算，但是不同振幅参数的传播子不能向量化计算。因此按照传播子类型+振幅参数的方式进行分组。
    定义：
    - 具有任务1中定义的相同“A传播子”和“B传播子”类型。
    - 具有相同的振幅名称（AMP的值）。

3. 我们需要参数值的列表，根据任务2的分组结果，输出组内参数的列表。
    定义：
    - 对于每个组合并组内的共振态的参数，输出参数的值列表。
    - 合并的方式为共振态之间相同的参数合并为一个参数，value值组合为列表。
    - 共振态Amplitude参数合并为二维list，例如[[a_resonance_const1, a_resonance_const2], [b_resonance_const1, b_resonance_const2]]。

输出格式：
{{
    "propagator_classification": {{
    "TypeA_TypeB": ["resonance_name_1"（共振名称1）, "resonance_name_2"（共振名称2）],
    "TypeA_TypeC": ["resonance_name_3"（共振名称3）]
    }},
    "amplitude_classification": {{
    "AMP_TypeA_TypeB": ["resonance_name_1"（共振名称1）, "resonance_name_2"（共振名称2）],
    "AMP_TypeA_TypeC": ["resonance_name_3"（共振名称3）]
    }},
    "parameter_lists": {{
    "AMP_TypeA_TypeB": ["mass":[0.980, 1.27], "width":[0.05, 0.15], "const":[0.5, 1.0], "theta":[0.0, 1.57]],
    "AMP_TypeA_TypeC": ["mass":[1.27], "width":[0.15], "const":[1.0], "theta":[1.57]
    }}
}}

"""
        return prompt
    
    def generate_load_data_prompt(self) -> str:
        """生成数据加载函数"""
        data_loading_section = self.sections.get('DATA_LOADING', '')
        sbc, amp = self.get_all_resonance_data()

        # 组装英文提示词
        prompt = f"""You are given a Python code template:
function template:
{data_loading_section}
```python
def load_data():
    data = {{}}
    
    # Load real data
    data['data_{{var}}'] = onp.load("data/real_data/{{var}}.npy")

    # Load MC data
    data['mc_{{var}}'] = onp.load("data/mc_truth/{{var}}.npy")

    # MC truth data (for constraints)
    data['truth_{{var}}'] = data['mc_{{var}}'][:, 0:150000]
    
    # Weight data
    try:
        data['wt_data_kk'] = onp.load("data/weight/weight_kk.npy")
    except FileNotFoundError:
        data['wt_data_kk'] = onp.ones_like(data['data_phi_kk'])
    
    return data

def normalize_data(data):
    # 计算归一化因子
    regular_{{var}} = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_{{var}}'])**2, axis=2)), axis=1
    )
    
    # 应用归一化
    data['data_{{var}}'] = onp.einsum("jkl,j->jkl", data['data_{{var}}'], regular_{{var}})
    data['mc_{{var}}'] = onp.einsum("jkl,j->jkl", data['mc_{{var}}'], regular_{{var}})
    data['truth_{{var}}'] = onp.einsum("jkl,j->jkl", data['truth_{{var}}'], regular_{{var}})
    
    return data

def prepare_data_for_jax(data, device=None):
    jax_data = {{}}
    for key, value in data.items():
        jax_data[key] = device_put(np.array(value), device=device)
    return jax_data
```

Task:
Replace {{var}} with each variable name from the given list.
For variables Sbc, the truth slicing is [0:150000], variables amp the truth slicing is [:, 0:150000].
For normalization, use "regular_{{var}}" as the normalization factor name and only amp need normalization.
Keep the rest of the code structure exactly the same.
Output the final Python code string only, without extra explanations.

Variable list:
sbc = {sbc}
amp = {amp}
"""
        return prompt
    
    def generate_calculate_function_prompt(self, ana_key, ana_value, resonance_name) -> str:
        """生成 calculate_{A_propagator_type}_{B_propagator_type} 函数的代码生成提示词"""
        all_resonances_info = self.config.get('resonances', {})
        resonance_info = json.dumps(all_resonances_info.get(resonance_name, {}), indent=4)
        prompt = f"""
### 1. 任务目标
你的任务是：
- 仅替换给定 Python 代码模板中的 placeholder 标记（如 {{calculation_name}}、{{A_propagator_param}} 等）。
- 保留模板中固定部分的结构、缩进、函数名、逻辑不变。
- 输出最终的完整 Python 函数代码字符串，不添加任何额外解释或注释。

---

### 2. 模板结构说明
- 固定部分：
  - 模板中的函数定义、缩进、固定逻辑、固定变量名。
  - 除 placeholder 外的所有代码必须原样保留。

- 可变部分（需要替换的 placeholder）：
  - {{calculation_name}}
  - {{A_propagator_param}} / {{B_propagator_param}}
  - {{A_propagator_type}} / {{B_propagator_type}}

---

### 3. 参数替换规则
1. calculation_name
   - 使用输入中提供的 ana_key 值。

2. A_propagator_param / B_propagator_param
   - 按模板中参数顺序，从 resonance 配置的 propagator 参数中提取变量名（不是值）。

3. A_propagator_type / B_propagator_type
   - 使用 resonance 配置中的 propagator_type。

4. 向量化方法选择规则（必须先判断再生成）
   - **条件 1**：如果该 propagator 的所有参数值都是固定值 → 使用直接函数调用：
     ```python
     A_propagator = BW({{A_propagator_param}})
     ```
   - **条件 2**：如果 Resonance_len_in_Group 为 False → 也使用直接函数调用（即使参数不是固定值）。
   - **条件 3**：其他情况 → 使用 vmap 方式：
     ```python
     A_propagator = np.moveaxis(
         vmap(partial({{A_propagator_type}}, Sbc={{A_propagator_param.Sbc}}))({{A_propagator_param}}), 1, 0
     )
     ```
     - {{A_propagator_param.Sbc}} 表示 Sbc 的变量名（如 phi_kk），不是值。
     - {{A_propagator_param}} 表示除 Sbc 外的参数变量名集合（如 A_mass, A_width）。

5. component_{{calculation_name}} 构造
   - 构造并生成 component_{{calculation_name}} 函数。
   - 与 calculate_{{calculation_name}} 相同，唯一不同是最后一步的 contraction 改为 `"ljk,lj->ljk"`。

---

### 4. 强制检测环节（生成前必须执行）
在生成代码前，必须逐项检查：
1. **A_propagator 检查**
   - 判断其所有参数是否为固定值。
   - 如果是 → 必须使用 direct call。
   - 如果 Resonance_len_in_Group 为 False → 必须使用 direct call。
   - 否则 → 使用 vmap。
2. **B_propagator 检查**
   - 同 A_propagator 的规则。
3. **参数格式**
   - 所有替换的参数必须符合规则（变量名而非值，顺序正确）。
4. **代码缩进**
   - 缩进与原模板完全一致。
5. **输出内容**
   - 仅输出最终 Python 函数代码字符串，不包含解释、注释或额外文本。
6. **规则违规处理**
   - 如果检测发现不符合规则，必须重新生成，直到符合为止。

---

function template:
```python
def calculate_{{calculation_name}}({{A_propagator_param}}, {{B_propagator_param}}, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = {{A_propagator_type}}({{A_propagator_param}})
    B_propagator = {{B_propagator_type}}({{B_propagator_param}})
    propagator_combined = dplex.deinsum("j, ij->ij", B_propagator, A_propagator)
    const_ph = dplex.dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex.deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex.deinsum("ljk,lj->jk", result, propagator_combined)
    return result
```

Resonance_Info:
{resonance_info}

calculation_name: {ana_key}
Resonance_len_in_Group: {len(ana_value) > 1}
"""

        return prompt

    def generate_extract_parameters_prompt(self, parameter_info) -> str:
        """生成 data_likelihood_{channel} 函数的代码生成提示词"""
        parameter_info_str = json.dumps(parameter_info, indent=4)
        prompt = f"""
### 1. 任务目标
你的任务是：
1. 根据输入的parameter信息生成args_list列表，填入参数的初始值。
2. 生成 extract_parameters 函数
    - 按照参数列表的顺序，从 args 中提取对应的值，并赋给相应的变量名。
    - 如果参数是二维数组，则在提取时保持二维数组的形状。

### 2. extract_parameters 模板
args_list = onp.array([按顺序填入全部参数值])

def extract_parameters(args):
    return {{
        "var_name_A": np.array(args[参数的序号]),
        "var_name_B": np.array(args[参数的序号]),
        # 如果是二维数组参数
        "var_name_C": np.array(args[起始序号:结束序号]).reshape(行数, 列数),
    }}
参数列表:
{parameter_info_str}

### 3. 强制检测环节（生成前必须执行）
在生成代码前，必须逐项检查：
1. **代码缩进**
   - 缩进与原模板完全一致。
2. **输出内容**
   - 仅输出最终 Python 函数代码字符串，不包含解释、注释或额外文本。
3. **规则违规处理**
   - 如果检测发现不符合规则，必须重新生成，直到符合为止。
"""
        return prompt
    
    def prepare_likelihood_info(self):
        pass 

    
    def generate_likelihood_function_prompt(self, parameter_info, resonance_calculation, extract_parameters) -> str:
        """生成 data_likelihood_{channel} 函数的代码生成提示词"""
        parameter_info_str = json.dumps(parameter_info["parameter_lists"], indent=4)
        likelihood_functions_section = self.sections.get('likelihood_functions', '')
        prompt = f"""
### 1. 任务目标
你的任务：
1. 输入信息中包含共振态信息和函数定义，以及一个要生成的函数的例子。首先看一下这些内容，并找到内在联系。
2. 理解函数例子中的代码的组织、联系、以及与函数定义中的函数之间的关系。
3. 理解共振态信息的层次，内容以及如何用这些共振态信息来生成一个由这些共振态组成的函数。
4. 将你理解的内容精炼的整理出来。
5. 根据你整理出来的内容和思路并根据输入信息生成函数，生成包括 data 与 mc 的两个似然函数，要求返回的内容只包含 python 代码字符串，不包含解释、注释或额外文本，缩进与函数例子一致。

### 2. 输入信息
共振态类型信息：
{parameter_info_str}

函数定义(calculate_xxx和component_xxx):
{resonance_calculation}
{extract_parameters}

函数例子：
{likelihood_functions_section}

"""
        return prompt

    def generate_partial_function(self, prompt: str, cache_file: str, check: bool) -> str:
        load_data_cache = self._load_cache(cache_file)
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        if prompt_hash in load_data_cache:
            print(f"✅ 发现匹配的缓存 {cache_file} (Hash: {prompt_hash[:8]}...)，从文件加载已生成的函数。")
            return load_data_cache[prompt_hash]['generated_code']

        print(f"📝 提示词长度: {len(prompt)} 字符")
        
        try:
            # 使用responses API调用Claude模型 (不使用messages参数)
            response = self.llm_client.responses(
                input_text=prompt,
                model=self.model
            )
            
            if not self.llm_client.validate_response(response):
                raise EasyTransError("LLM响应验证失败")
            
            generated_code = self.llm_client.extract_content(response)
            print(generated_code)
            
            if not generated_code:
                raise EasyTransError("LLM返回空的代码内容")
            
            print(f"✨ 函数生成成功！代码长度: {len(generated_code)} 字符")

            check_prompt = f"""{prompt}
Please strictly check the following code for compliance with all the rules mentioned in the prompt. If any rule is violated, regenerate the code until it fully complies with all the rules.
Code:
{generated_code}
"""
            if check:
                check_out = self.llm_client.responses(
                    input_text=check_prompt,
                    model=self.model_check
                )
                generated_code = self.llm_client.extract_content(check_out)
                print(f"🔍 代码检查结果: {generated_code}")

            load_data_cache[prompt_hash] = {
                "prompt": prompt,
                "generated_code": generated_code
            }
            self._save_cache(load_data_cache, cache_file)

            return generated_code
            
        except Exception as e:
            print(f"❌ LLM代码生成失败: {e}")
            raise
    
    def generate_complete_resonance_functions(self) -> Dict[str, str]:
        """生成指定共振态的完整函数集合"""
        print(f"🚀 开始生成共振态的完整函数集合...")

        functions = {}

        try:
            ana_prompt = self.analysis_toml_config_prompt()
            ana_result = self.generate_partial_function(ana_prompt, "agent/cache/ana_cache.json", False)
            ana_result = json.loads(ana_result)
            print("ana_reuslt:",ana_result)
        except Exception as e:
            print(f"⚠️  分析失败: {e}")
        time.sleep(1)

        try:
            data_load_prompt = self.generate_load_data_prompt()
            functions['data_load'] = self.generate_partial_function(data_load_prompt, "agent/cache/load_data_cache.json", False)
        except Exception as e:
            print(f"⚠️  data load 函数生成失败: {e}")
        time.sleep(1) 

        try:
            resonance_calculation_functions = []
            for ana_key, ana_value in ana_result["propagator_classification"].items():
                ana_value = sorted(ana_value)
                resonance_name = ana_value[0]
                resonance_calculation_prompt = self.generate_calculate_function_prompt(ana_key,ana_value,resonance_name)
                resonance_calculation_functions.append(self.generate_partial_function(resonance_calculation_prompt, f"agent/cache/resonance_calculation_{resonance_name}.json", True))
            functions['resonance_calculation'] = "\n\n".join(resonance_calculation_functions)
        except Exception as e:
            print(f"⚠️  calculation 函数生成失败: {e}")
        time.sleep(1)

        try:
            extract_parameters_prompt = self.generate_extract_parameters_prompt(ana_result["parameter_lists"])
            functions['extract_parameters'] = self.generate_partial_function(extract_parameters_prompt, "agent/cache/extract_parameters_cache.json", False)
        except Exception as e:
            print(f"⚠️  extract_parameters 函数生成失败: {e}")
        time.sleep(1)

        try:
            likelihood_function_prompt = self.generate_likelihood_function_prompt(ana_result,resonance_calculation_functions,functions["extract_parameters"])
            functions['likelihood_function'] = self.generate_partial_function(likelihood_function_prompt, "agent/cache/analysis_likelihood_cache.json", False)
        except Exception as e:
            print(f"⚠️  likelihood 函数生成失败: {e}")
        time.sleep(1) 
        
        print(f"✅ 函数集合生成完成！")
        return functions
    
    def save_generated_functions(self, functions: Dict[str, str],
                               output_path: str = None) -> str:
        """保存生成的函数到文件"""
        if output_path is None:
            output_path = f"agent/generated_script.py"
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        

        common_utilities_section = self.sections.get('COMMON_UTILITIES', '')
        path_config_section = self.sections.get('PATH_CONFIG', '')
        logging_config_section = self.sections.get('LOGGING_CONFIG', '')
        physics_functions_section = self.sections.get('PHYSICS_FUNCTIONS', '')
        combined_likelihood_section = self.sections.get('combined_likelihood_function', '')
        
        # 组合头部
        header = f"""# Auto-generated by LLMResonanceGenerator
# Do not edit manually!
{common_utilities_section}
{path_config_section}
{logging_config_section}
{physics_functions_section}
"""
        # 组合所有函数
        full_code = header

        if 'data_load' in functions:
            full_code += f"\n\n"
            full_code += functions['data_load']
        
        if 'resonance_calculation' in functions:
            full_code += f"\n\n"
            full_code += functions['resonance_calculation']

        if 'extract_parameters' in functions:
            full_code += f"\n\n"
            full_code += functions['extract_parameters']

        if 'likelihood_function' in functions:
            full_code += f"\n\n"
            full_code += functions['likelihood_function'] + "\n\n" + combined_likelihood_section
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        print(f"💾 生成的函数已保存到: {output_path}")
        return output_path
    

def main():
    """主函数 - 演示LLM驱动的代码生成"""
    print("🤖 LLM驱动的PWA共振态函数生成器")
    print("=" * 50)
    
    try:
        # 创建生成器
        generator = LLMResonanceGenerator(model="gpt-5-mini-2025-08-07", model_check="gpt-5-2025-08-07")
        
        # 打印配置摘要
        generator.print_config_summary()
        
        # 获取所有共振态名称
        resonance_names = generator.get_all_resonance_names()
        print(f"📊 全部共振态: {resonance_names}")
        
        functions = generator.generate_complete_resonance_functions()
        generator.save_generated_functions(functions)
        
    except Exception as e:
        print(f"❌ 主程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()