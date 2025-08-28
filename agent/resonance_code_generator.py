#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共振态代码生成器
读取 TOML 配置文件，分析需要生成的共振态计算函数
"""

import toml
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


@dataclass
class ResonanceInfo:
    """共振态信息数据类"""
    name: str
    type: str
    mod_name: str
    amplitude_wave: str
    propagators: Dict[str, str]
    parameters: Dict[str, Any]
    coefficients: Dict[str, Any]
    function_template: str
    lasso_function_template: str
    parameter_extraction: List[str]
    multi_component: Optional[int] = None


@dataclass
class AmplitudeCalculation:
    """振幅计算信息"""
    name: str
    resonance: str
    data_source: str
    calculation_type: str  # "data_likelihood", "mc_likelihood", "constraint"


class ResonanceConfigParser:
    """共振态配置解析器"""
    
    def __init__(self, config_path: str):
        """
        初始化解析器
        
        Args:
            config_path: TOML 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.resonances = {}
        self.amplitude_calculations = []
        
    def _load_config(self) -> Dict[str, Any]:
        """加载 TOML 配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = toml.load(f)
            print(f"✓ 成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            raise RuntimeError(f"无法加载配置文件 {self.config_path}: {e}")
    
    def parse_resonances(self) -> Dict[str, ResonanceInfo]:
        """解析共振态配置"""
        resonances_config = self.config.get('resonances', {})
        
        for resonance_name, resonance_data in resonances_config.items():
            # 处理普通共振态
            if resonance_name != 'f2_group':
                self.resonances[resonance_name] = self._parse_single_resonance(
                    resonance_name, resonance_data
                )
            # 处理 f2 共振态组（特殊情况）
            else:
                self.resonances[resonance_name] = self._parse_f2_group(
                    resonance_name, resonance_data
                )
        
        print(f"✓ 解析了 {len(self.resonances)} 个共振态配置")
        return self.resonances
    
    def _parse_single_resonance(self, name: str, data: Dict[str, Any]) -> ResonanceInfo:
        """解析单个共振态配置"""
        code_gen = data.get('code_generation', {})
        
        return ResonanceInfo(
            name=data.get('name', name),
            type=data.get('type', ''),
            mod_name=data.get('mod_name', ''),
            amplitude_wave=data.get('amplitude_wave', ''),
            propagators=data.get('propagators', {}),
            parameters=data.get('parameters', {}),
            coefficients=data.get('coefficients', {}),
            function_template=code_gen.get('function_template', ''),
            lasso_function_template=code_gen.get('lasso_function_template', ''),
            parameter_extraction=code_gen.get('parameter_extraction', []),
            multi_component=code_gen.get('multi_component')
        )
    
    def _parse_f2_group(self, name: str, data: Dict[str, Any]) -> ResonanceInfo:
        """解析 f2 共振态组配置"""
        code_gen = data.get('code_generation', {})
        
        # 合并所有 f2 子共振态的参数
        all_parameters = {}
        all_coefficients = {}
        
        for f2_name in ['f2_1525', 'f2_2150', 'f2_2340']:
            if f2_name in data:
                f2_data = data[f2_name]
                all_parameters.update({f"{f2_name}_{k}": v for k, v in f2_data.get('parameters', {}).items()})
                all_coefficients.update({f"{f2_name}_{k}": v for k, v in f2_data.get('coefficients', {}).items()})
        
        return ResonanceInfo(
            name=data.get('name', name),
            type=data.get('type', ''),
            mod_name=str(data.get('mod_names', [])),
            amplitude_wave=data.get('amplitude_wave', ''),
            propagators=data.get('propagators', {}),
            parameters=all_parameters,
            coefficients=all_coefficients,
            function_template=code_gen.get('function_template', ''),
            lasso_function_template=code_gen.get('lasso_function_template', ''),
            parameter_extraction=code_gen.get('parameter_extraction', []),
            multi_component=data.get('n_components')
        )
    
    def parse_amplitude_calculations(self) -> List[AmplitudeCalculation]:
        """解析振幅计算配置"""
        amplitude_config = self.config.get('amplitude_calculations', {})
        
        for calc_type in ['data_likelihood', 'mc_likelihood', 'constraint']:
            for calc_data in amplitude_config.get(calc_type, []):
                self.amplitude_calculations.append(
                    AmplitudeCalculation(
                        name=calc_data.get('name', ''),
                        resonance=calc_data.get('resonance', ''),
                        data_source=calc_data.get('data_source', ''),
                        calculation_type=calc_type
                    )
                )
        
        print(f"✓ 解析了 {len(self.amplitude_calculations)} 个振幅计算配置")
        return self.amplitude_calculations


class ResonanceFunctionAnalyzer:
    """共振态函数分析器"""
    
    def __init__(self, resonances: Dict[str, ResonanceInfo], 
                 amplitude_calculations: List[AmplitudeCalculation]):
        self.resonances = resonances
        self.amplitude_calculations = amplitude_calculations
        
    def analyze_required_functions(self) -> Dict[str, Any]:
        """分析需要生成的共振态函数"""
        
        analysis = {
            'basic_functions': self._analyze_basic_functions(),
            'composite_functions': self._analyze_composite_functions(),
            'lasso_functions': self._analyze_lasso_functions(),
            'parameter_extraction': self._analyze_parameter_extraction(),
            'amplitude_calculations': self._analyze_amplitude_calculations()
        }
        
        return analysis
    
    def _analyze_basic_functions(self) -> List[Dict[str, Any]]:
        """分析基础物理函数需求"""
        basic_functions = []
        
        # 从共振态配置中提取需要的基础函数
        propagator_types = set()
        
        for resonance in self.resonances.values():
            for prop_name, prop_type in resonance.propagators.items():
                propagator_types.add(prop_type)
        
        function_mapping = {
            'BW': 'BW(m_, w_, Sbc) - Breit-Wigner 共振态形状',
            'flatte980': 'flatte980(m_, g_pipi, rg, Sbc) - f980 Flatte 形状',
            'flatte1270': 'flatte1270(m_, w_, Sbc) - f1270 Flatte 形状'
        }
        
        for prop_type in propagator_types:
            if prop_type in function_mapping:
                basic_functions.append({
                    'name': prop_type,
                    'description': function_mapping[prop_type],
                    'required': True
                })
        
        return basic_functions
    
    def _analyze_composite_functions(self) -> List[Dict[str, Any]]:
        """分析复合共振态计算函数需求"""
        composite_functions = []
        
        for resonance in self.resonances.values():
            if resonance.function_template:
                composite_functions.append({
                    'name': resonance.function_template,
                    'resonance': resonance.name,
                    'type': resonance.type,
                    'amplitude_wave': resonance.amplitude_wave,
                    'propagators': resonance.propagators,
                    'parameters': list(resonance.parameters.keys()),
                    'coefficients': list(resonance.coefficients.keys()),
                    'parameter_extraction': resonance.parameter_extraction
                })
        
        return composite_functions
    
    def _analyze_lasso_functions(self) -> List[Dict[str, Any]]:
        """分析 Lasso 约束函数需求"""
        lasso_functions = []
        
        for resonance in self.resonances.values():
            if resonance.lasso_function_template:
                lasso_functions.append({
                    'name': resonance.lasso_function_template,
                    'resonance': resonance.name,
                    'base_function': resonance.function_template,
                    'description': f"Lasso 版本的 {resonance.name} 计算（保留 l 维度用于约束）"
                })
        
        return lasso_functions
    
    def _analyze_parameter_extraction(self) -> Dict[str, Any]:
        """分析参数提取需求"""
        all_parameters = []
        parameter_indices = {}
        
        # 收集所有参数
        fixed_params = self.resonances.get('fixed_parameters', {})
        
        current_index = 0
        for param_name, param_value in fixed_params.items():
            all_parameters.append({
                'name': param_name,
                'value': param_value,
                'index': current_index,
                'fixed': True
            })
            parameter_indices[param_name] = current_index
            current_index += 1
        
        # 添加浮动参数
        for resonance in self.resonances.values():
            for param_name, param_info in resonance.parameters.items():
                if not param_info.get('fixed', False):
                    all_parameters.append({
                        'name': f"{resonance.name}_{param_name}",
                        'value': param_info.get('value'),
                        'index': current_index,
                        'fixed': False,
                        'resonance': resonance.name
                    })
                    parameter_indices[f"{resonance.name}_{param_name}"] = current_index
                    current_index += 1
            
            # 添加系数参数
            for coeff_name, coeff_info in resonance.coefficients.items():
                if not coeff_info.get('fixed', False):
                    all_parameters.append({
                        'name': f"{resonance.name}_{coeff_name}",
                        'value': coeff_info.get('value'),
                        'index': current_index,
                        'fixed': False,
                        'resonance': resonance.name
                    })
                    parameter_indices[f"{resonance.name}_{coeff_name}"] = current_index
                    current_index += 1
        
        return {
            'total_parameters': len(all_parameters),
            'fixed_parameters': len([p for p in all_parameters if p['fixed']]),
            'float_parameters': len([p for p in all_parameters if not p['fixed']]),
            'parameters': all_parameters,
            'parameter_indices': parameter_indices
        }
    
    def _analyze_amplitude_calculations(self) -> Dict[str, List[Dict[str, Any]]]:
        """分析振幅计算需求"""
        calculations = {
            'data_likelihood': [],
            'mc_likelihood': [],
            'constraint': []
        }
        
        for calc in self.amplitude_calculations:
            resonance_info = self.resonances.get(calc.resonance, None)
            if resonance_info:
                calculations[calc.calculation_type].append({
                    'name': calc.name,
                    'resonance': calc.resonance,
                    'data_source': calc.data_source,
                    'function_template': resonance_info.function_template,
                    'lasso_function_template': resonance_info.lasso_function_template
                })
        
        return calculations


def main():
    """主函数：演示代码生成器的使用"""
    
    print("=== PWA 共振态代码生成器 ===\n")
    
    # 1. 解析配置文件
    config_path = "/home/sean/PWACG/agent/resonances_config.toml"
    parser = ResonanceConfigParser(config_path)
    
    # 2. 解析共振态和振幅计算
    resonances = parser.parse_resonances()
    amplitude_calculations = parser.parse_amplitude_calculations()
    
    print(f"\n=== 配置文件解析结果 ===")
    print(f"共振态数量: {len(resonances)}")
    print(f"振幅计算数量: {len(amplitude_calculations)}")
    
    # 3. 分析需要生成的函数
    analyzer = ResonanceFunctionAnalyzer(resonances, amplitude_calculations)
    analysis_result = analyzer.analyze_required_functions()
    
    # 4. 输出分析结果
    print(f"\n=== 需要生成的函数分析 ===")
    
    print(f"\n1. 基础物理函数 ({len(analysis_result['basic_functions'])} 个):")
    for func in analysis_result['basic_functions']:
        print(f"   - {func['description']}")
    
    print(f"\n2. 复合共振态计算函数 ({len(analysis_result['composite_functions'])} 个):")
    for func in analysis_result['composite_functions']:
        print(f"   - {func['name']}(): {func['resonance']} ({func['type']}) -> {func['amplitude_wave']}")
        print(f"     传播子: {func['propagators']}")
        print(f"     参数: {func['parameter_extraction']}")
    
    print(f"\n3. Lasso 约束函数 ({len(analysis_result['lasso_functions'])} 个):")
    for func in analysis_result['lasso_functions']:
        print(f"   - {func['name']}(): {func['description']}")
    
    print(f"\n4. 参数提取配置:")
    param_info = analysis_result['parameter_extraction']
    print(f"   - 总参数数量: {param_info['total_parameters']}")
    print(f"   - 固定参数: {param_info['fixed_parameters']}")
    print(f"   - 浮动参数: {param_info['float_parameters']}")
    
    print(f"\n5. 振幅计算配置:")
    amp_calcs = analysis_result['amplitude_calculations']
    for calc_type, calcs in amp_calcs.items():
        print(f"   - {calc_type}: {len(calcs)} 个计算")
        for calc in calcs:
            print(f"     * {calc['name']} ({calc['resonance']} -> {calc['data_source']})")
    
    # 5. 保存分析结果到 JSON 文件
    output_path = "/home/sean/PWACG/agent/resonance_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 分析结果已保存到: {output_path}")
    
    # 6. 输出代码生成建议
    print(f"\n=== 代码生成建议 ===")
    print("基于分析结果，建议按以下顺序生成代码:")
    print("1. 确保基础物理函数 (BW, flatte980, flatte1270) 存在")
    print("2. 生成复合共振态计算函数")
    print("3. 生成对应的 Lasso 约束版本函数")
    print("4. 生成参数提取函数 extract_parameters()")
    print("5. 生成似然计算类中的振幅计算代码")
    print("6. 整合所有组件到完整的拟合脚本")


if __name__ == "__main__":
    main()