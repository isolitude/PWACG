#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM驱动的PWA共振态函数生成器 (使用EasyTrans API)
读取 resonances_config.toml 配置，使用大模型API生成共振态计算函数

特点:
- 使用 EasyTrans API 作为 LLM 引擎
- TOML 配置作为"燃料"，驱动代码生成
- 智能分析共振态物理参数，生成相应的JAX函数
- 支持多种共振态类型 (BW, Flatte, 等)

作者: LLM助手
日期: 2025-01-21
"""

import os
import sys
import toml
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目路径
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

# 导入EasyTrans客户端
from agent.easytrans_client import EasyTransClient, EasyTransError


class LLMResonanceGenerator:
    """LLM驱动的共振态代码生成器"""
    
    def __init__(self, config_path: str = "agent/resonances_config.toml", model: str = "claude-opus-4-20250514"):
        """
        初始化LLM代码生成器
        
        Args:
            config_path: TOML配置文件路径
            model: 使用的LLM模型
        """
        self.config_path = config_path
        self.model = model
        self.config = self.load_config()
        
        # 初始化EasyTrans客户端
        self.llm_client = EasyTransClient()
        print(f"🤖 LLM引擎初始化完成: {model}")
        
        # System prompt - specialized for generating PWA resonance functions
        self.system_prompt = """You are a professional physics computation code generator, specializing in Partial Wave Analysis (PWA) in particle physics.

Your task is to generate high-quality JAX Python functions for calculating resonance contributions based on TOML configuration files.

Key requirements:
1. Use JAX numpy (import jax.numpy as np) for numerical computations
2. Use dplex library for complex number operations (from dlib import dplex)
3. Use vmap and partial for vectorized computations (from jax import vmap; from functools import partial)
4. Strictly implement resonance propagators according to physics formulas
5. Correctly handle complex coefficient construction and application
6. Follow existing code naming conventions and structure
7. Add concise but meaningful physics comments
8. Ensure numerical stability of functions

Supported resonance types:
- BW: Breit-Wigner resonance
- flatte: Flatte shape resonance  
- flatte980: f980 special Flatte shape
- flatte1270: f1270 special Flatte shape

Generate complete, runnable Python functions without overly detailed docstrings."""
        
    def load_config(self) -> Dict[str, Any]:
        """加载TOML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = toml.load(f)
            print(f"✅ 配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            raise ValueError(f"❌ 无法加载配置文件 {self.config_path}: {e}")
    
    def extract_resonance_config(self, resonance_name: str) -> Dict[str, Any]:
        """提取指定共振态的配置"""
        if 'resonances' not in self.config:
            raise ValueError("配置文件中未找到resonances节")
        
        if resonance_name not in self.config['resonances']:
            raise ValueError(f"配置文件中未找到共振态: {resonance_name}")
        
        resonance_config = self.config['resonances'][resonance_name]
        fixed_params = self.config.get('fixed_parameters', {})
        
        return {
            'name': resonance_config['name'],
            'type': resonance_config['type'],
            'config': resonance_config,
            'fixed_parameters': fixed_params
        }
    
    def create_generation_prompt(self, resonance_name: str, function_type: str = "standard") -> str:
        """
        Create LLM code generation prompt in English
        
        Args:
            resonance_name: Resonance name (e.g. 'f980')
            function_type: Function type ('standard' or 'lasso')
        """
        resonance_config = self.extract_resonance_config(resonance_name)
        config = resonance_config['config']
        
        # Build detailed prompt in English
        prompt = f"""Generate a PWA resonance calculation function based on the following TOML configuration.

## Resonance Configuration:
- **Resonance Name**: {resonance_config['name']}
- **Type**: {resonance_config['type']}
- **Mode Identifier**: {config['mod_name']}
- **Amplitude Wave**: {config['amplitude_wave']}

## Propagator Configuration:
"""
        
        for prop_name, prop_type in config['propagators'].items():
            prompt += f"- **{prop_name}**: {prop_type}\n"
        
        prompt += "\n## Physical Parameters:\n"
        for param_name, param_info in config['parameters'].items():
            prompt += f"- **{param_name}**: {param_info['value']:.6f}"
            if 'error' in param_info:
                prompt += f" ± {param_info['error']:.6f}"
            if 'range' in param_info:
                prompt += f" (range: {param_info['range']})"
            prompt += f" {'(fixed)' if param_info.get('fixed', False) else '(floating)'}\n"
        
        prompt += "\n## Complex Coefficients:\n"
        for coeff_name, coeff_info in config['coefficients'].items():
            prompt += f"- **{coeff_name}**: {coeff_info['value']:.6f}"
            if 'error' in coeff_info:
                prompt += f" ± {coeff_info['error']:.6f}"
            prompt += f" {'(fixed)' if coeff_info.get('fixed', False) else '(floating)'}\n"
        
        prompt += f"\n## Fixed Parameters:\n"
        for param_name, param_value in resonance_config['fixed_parameters'].items():
            prompt += f"- **{param_name}**: {param_value}\n"
        
        # Add specific requirements based on function type
        if function_type == "standard":
            prompt += f"""
## Code Generation Requirements:

Generate function: `calculate_BW_{resonance_config['type']}`

Function signature should include:
- phi_mass, phi_width: Phi meson parameters
- {resonance_name} related mass, width/coupling parameters
- Complex coefficient arrays (const, theta)
- Data arrays (phi_kk, f_kk, phif0_kk/phif2_kk)

The function should:
1. Calculate Phi meson BW propagator
2. Calculate {resonance_name} {resonance_config['type']} propagator
3. Combine both propagators
4. Construct complex coefficients
5. Apply to amplitude data
6. Return final amplitude (dimensions: [j, k])

Follow JAX+dplex programming patterns strictly, ensuring numerical correctness.
"""
        else:  # lasso version
            prompt += f"""
## Code Generation Requirements:

Generate function: `lasso_calculate_BW_{resonance_config['type']}`

This is a special version for constraint calculations. Difference from standard version:
- Final Einstein summation preserves l dimension: ljk,lj->ljk (instead of ljk,lj->jk)
- Return dimensions: [l, j, k] (preserve l dimension for branching ratio constraint calculation)
- Other calculation logic same as standard version

This function is mainly used for calculating resonance branching ratio constraints, ensuring the total contribution of each resonance meets physical expectations.
"""
        
        prompt += """
## Output Format:
Please output complete Python function code directly, including:
1. Function definition
2. Concise docstring with physics meaning
3. Complete function body implementation
4. Necessary comments

Do not include any explanatory text, only pure code.
"""
        
        return prompt
    
    def generate_function_with_llm(self, resonance_name: str, function_type: str = "standard") -> str:
        """
        使用LLM生成共振态函数
        
        Args:
            resonance_name: 共振态名称
            function_type: 函数类型 ('standard' 或 'lasso')
            
        Returns:
            生成的函数代码
        """
        prompt = self.create_generation_prompt(resonance_name, function_type)
        
        print(f"🧠 正在使用 {self.model} 生成 {resonance_name} 的 {function_type} 函数...")
        print(f"📝 提示词长度: {len(prompt)} 字符")
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # 使用messages API调用Claude模型
            response = self.llm_client.messages(
                messages=messages,
                model=self.model,
                max_tokens=4000000,
                system=self.system_prompt
            )
            
            if not self.llm_client.validate_response(response):
                raise EasyTransError("LLM响应验证失败")
            
            generated_code = self.llm_client.extract_content(response)
            
            if not generated_code:
                raise EasyTransError("LLM返回空的代码内容")
            
            print(f"✨ 函数生成成功！代码长度: {len(generated_code)} 字符")
            return generated_code
            
        except Exception as e:
            print(f"❌ LLM代码生成失败: {e}")
            raise
    
    def generate_parameter_extraction_with_llm(self, resonance_name: str) -> str:
        """Generate parameter extraction code using LLM"""
        resonance_config = self.extract_resonance_config(resonance_name)
        
        prompt = f"""Generate parameter extraction code for resonance {resonance_name}.

## Resonance Configuration:
{json.dumps(resonance_config, indent=2, ensure_ascii=False)}

## Requirements:
Generate code to extract parameters from args array, following this format:

```python
# === {resonance_name} resonance parameter extraction ===
# Based on config: {resonance_config['name']} ({resonance_config['type']})

# Fixed parameters
phi_mass = np.array([1.02])
phi_width = np.array([0.004])

# {resonance_name} physical parameters
# Extract from corresponding positions in args array, add comments explaining parameter meaning and values

# Complex coefficients
# Distinguish between fixed and floating coefficients, construct correct array shapes
```

Output only code, no other explanations.
"""
        
        print(f"🧠 正在生成 {resonance_name} 的参数提取代码...")
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm_client.messages(
                messages=messages,
                model=self.model,
                max_tokens=4000000,
                system=self.system_prompt
            )
            
            generated_code = self.llm_client.extract_content(response)
            print(f"✨ 参数提取代码生成成功！")
            return generated_code
            
        except Exception as e:
            print(f"❌ 参数提取代码生成失败: {e}")
            raise
    
    def generate_complete_resonance_functions(self, resonance_name: str) -> Dict[str, str]:
        """生成指定共振态的完整函数集合"""
        print(f"🚀 开始生成 {resonance_name} 共振态的完整函数集合...")
        
        functions = {}
        
        # 生成标准计算函数
        try:
            functions['standard'] = self.generate_function_with_llm(resonance_name, "standard")
        except Exception as e:
            print(f"⚠️  标准函数生成失败: {e}")
            functions['standard'] = f"# 标准函数生成失败: {e}"
        
        time.sleep(1)  # 避免API限制
        
        # 生成lasso版本函数
        try:
            functions['lasso'] = self.generate_function_with_llm(resonance_name, "lasso")
        except Exception as e:
            print(f"⚠️  Lasso函数生成失败: {e}")
            functions['lasso'] = f"# Lasso函数生成失败: {e}"
        
        time.sleep(1)  # 避免API限制
        
        # 生成参数提取代码
        try:
            functions['parameter_extraction'] = self.generate_parameter_extraction_with_llm(resonance_name)
        except Exception as e:
            print(f"⚠️  参数提取代码生成失败: {e}")
            functions['parameter_extraction'] = f"# 参数提取代码生成失败: {e}"
        
        print(f"✅ {resonance_name} 函数集合生成完成！")
        return functions
    
    def save_generated_functions(self, functions: Dict[str, str], resonance_name: str, 
                               output_path: str = None) -> str:
        """保存生成的函数到文件"""
        if output_path is None:
            output_path = f"agent/llm_generated_{resonance_name}_functions.py"
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成文件头部
        header = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM自动生成的{resonance_name}共振态计算函数
基于配置文件: {self.config_path}
生成模型: {self.model}
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

警告：此文件由LLM自动生成，手动修改可能会被覆盖
"""

import jax.numpy as np
from jax import vmap
from functools import partial
from dlib import dplex

# 依赖的基础函数 (需要从其他模块导入)
# from fit_hvp_templates import BW, flatte980, flatte1270

'''
        
        # 组合所有函数
        full_code = header
        
        if 'parameter_extraction' in functions:
            full_code += f"\n\n# 参数提取代码\n"
            full_code += functions['parameter_extraction']
        
        if 'standard' in functions:
            full_code += f"\n\n# 标准计算函数\n"
            full_code += functions['standard']
        
        if 'lasso' in functions:
            full_code += f"\n\n# Lasso版本函数 (用于约束计算)\n"
            full_code += functions['lasso']
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        print(f"💾 生成的函数已保存到: {output_path}")
        return output_path
    
    def demonstrate_generation(self, resonance_name: str = "f980"):
        """演示完整的代码生成流程"""
        print(f"🎭 演示 {resonance_name} 共振态函数生成流程")
        print("="*60)
        
        try:
            # 分析配置
            config = self.extract_resonance_config(resonance_name)
            print(f"📊 共振态分析:")
            print(f"   - 名称: {config['name']}")
            print(f"   - 类型: {config['type']}")
            print(f"   - 传播子: {config['config']['propagators']}")
            
            # 生成函数
            functions = self.generate_complete_resonance_functions(resonance_name)
            
            # 保存文件
            output_file = self.save_generated_functions(functions, resonance_name)
            
            # 统计信息
            total_lines = sum(func_code.count('\n') for func_code in functions.values())
            print(f"\n📈 生成统计:")
            print(f"   - 生成函数数量: {len(functions)}")
            print(f"   - 总代码行数: {total_lines}")
            print(f"   - 输出文件: {output_file}")
            
            print(f"\n🎉 {resonance_name} 共振态函数生成完成！")
            
        except Exception as e:
            print(f"❌ 生成流程失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数 - 演示LLM驱动的代码生成"""
    print("🤖 LLM驱动的PWA共振态函数生成器")
    print("=" * 50)
    
    try:
        # 创建生成器 (使用Claude模型，特别适合代码生成)
        generator = LLMResonanceGenerator(model="claude-opus-4-20250514")
        
        # 演示f980共振态函数生成
        generator.demonstrate_generation("f980")
        
        print("\n" + "="*50)
        print("🎯 生成完成！你可以:")
        print("1. 检查生成的代码文件")
        print("2. 将函数集成到现有项目中")
        print("3. 运行测试验证函数正确性")
        print("4. 根据需要调整LLM提示词来优化生成结果")
        
    except Exception as e:
        print(f"❌ 主程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()