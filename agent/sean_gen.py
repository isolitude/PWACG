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


class LLMResonanceGenerator:
    """LLM驱动的共振态代码生成器"""
    
    def __init__(self, config_path: str = "agent/resonances_config.toml", model: str = "o3-pro-2025-06-10"):
        """
        初始化LLM代码生成器
        
        Args:
            config_path: TOML配置文件路径
            model: 使用的LLM模型
        """
        # 初始化EasyTrans客户端
        self.llm_client = EasyTransClient()
        print(f"🤖 LLM引擎初始化完成: {model}")
        
        self.config_path = config_path
        self.model = model
        self.config = self.load_config()

        # 解析模板部分
        self.sections = parse_template_sections("agent/common_template.py")
        
        self.system_prompt = """You are a professional physics computation code generator, specializing in Partial Wave Analysis (PWA) in particle physics.Generate complete, runnable Python functions without overly detailed docstrings."""
        
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
            prop["Sbc"]["name"]
            for resonance in self.config["resonances"].values()
            for prop in resonance.get("propagators", {}).values()
            if "Sbc" in prop
        ]
        # 提取所有 AMP
        amp_list = [
            resonance["coefficients"]["AMP"]["name"]
            for resonance in self.config["resonances"].values()
            if "AMP" in resonance.get("coefficients", {})
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
            coefficients = config.get('coefficients', {})
            const_count = len([k for k in coefficients.keys() if k.startswith('const')])
            theta_count = len([k for k in coefficients.keys() if k.startswith('theta')])
            print(f"   - 系数: {const_count} const, {theta_count} theta")
            
            print()
    
    def create_generation_prompt(self, resonance_name: str, function_type: str = "standard") -> str:
        prompt = f"# 生成 {resonance_name} 共振态的 {function_type} 函数\n"        
        return prompt
    
    def generate_load_data_function(self) -> str:
        """生成数据加载函数"""
        data_loading_section = self.sections.get('DATA_LOADING', '')
        sbc, amp = self.get_all_resonance_data()

        # 组装英文提示词
        prompt = f"""You are given a Python code template:
```python
{data_loading_section}
```
Task:
        
Replace {{var}} with each variable name from the given list.
For variables Sbc, the truth slicing is [0:150000], variables amp the truth slicing is [:, 0:150000].
For normalization, use "regular_{{var}}" as the normalization factor name and only amp need normalization.
Keep the rest of the code structure exactly the same.
Output the final Python code block only, without extra explanations.

Variable list:
sbc = {sbc}
amp = {amp}
"""

        print(f"🧠 正在使用 {self.model} 生成 load_data 函数...")
        print(prompt)
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
            return generated_code
            
        except Exception as e:
            print(f"❌ LLM代码生成失败: {e}")
            raise

    
    def generate_function_with_llm(self, resonance_name: str, function_type: str = "standard") -> str:
        """
        使用LLM生成共振态函数
        
        Args:
            resonance_name: 共振态名称
            function_type: 函数类型 ('standard' 或 'lasso')
            
        Returns:
            生成的函数代码
        """

        # 从模板中读取相关部分
        physics_functions = self.sections.get('PHYSICS_FUNCTIONS', '')
        
        # 获取共振态配置
        resonance_config = self.extract_resonance_config(resonance_name)
        
        # 构建提示词
        prompt = self.create_generation_prompt(resonance_name, function_type)
        prompt += f"\n配置信息:\n{json.dumps(resonance_config, indent=2, ensure_ascii=False)}\n"
        prompt += f"\n可用的物理函数模板:\n{physics_functions}\n"


        print(f"🧠 正在使用 {self.model} 生成 {resonance_name} 的 {function_type} 函数...")
        print(f"📝 提示词长度: {len(prompt)} 字符")
        
        try:
            # 使用responses API调用Claude模型 (不使用messages参数)
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            response = self.llm_client.responses(
                input_text=full_prompt,
                model=self.model
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
        
        print(f"✅ {resonance_name} 函数集合生成完成！")
        return functions
    
    def save_generated_functions(self, functions: Dict[str, str], resonance_name: str, 
                               output_path: str = None) -> str:
        """保存生成的函数到文件"""
        if output_path is None:
            output_path = f"agent/llm_generated_{resonance_name}_functions.py"
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建文件头部
        header = ""

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
    

def main():
    """主函数 - 演示LLM驱动的代码生成"""
    print("🤖 LLM驱动的PWA共振态函数生成器")
    print("=" * 50)
    
    try:
        # 创建生成器
        generator = LLMResonanceGenerator(model="gpt-5-2025-08-07")
        
        # 打印配置摘要
        generator.print_config_summary()
        
        # 获取所有共振态名称
        resonance_names = generator.get_all_resonance_names()
        print(f"📊 可用共振态: {resonance_names}")
        
        # 测试提取特定共振态配置
        if resonance_names:
            test_resonance = resonance_names[0]
            print(f"\n🔬 测试提取配置: {test_resonance}")
            config = generator.extract_resonance_config(test_resonance)
            print(f"配置结构: {json.dumps(config, indent=2, ensure_ascii=False)}")
        
        generator.generate_load_data_function()
        # generator.generate_complete_resonance_functions(test_resonance)
        
    except Exception as e:
        print(f"❌ 主程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()