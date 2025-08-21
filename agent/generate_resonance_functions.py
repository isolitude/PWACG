#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM驱动的共振态函数生成器
从 resonances_config.toml 读取配置，自动生成 PWA 共振态计算函数

作者: LLM助手
日期: 2025-01-21
"""

import os
import sys
import toml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

class ResonanceCodeGenerator:
    """共振态代码生成器"""
    
    def __init__(self, config_path: str = "agent/resonances_config.toml"):
        """
        初始化代码生成器
        
        Args:
            config_path: TOML配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.generated_functions = []
        
    def load_config(self) -> Dict[str, Any]:
        """加载TOML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except Exception as e:
            raise ValueError(f"无法加载配置文件 {self.config_path}: {e}")
    
    def extract_f980_config(self) -> Dict[str, Any]:
        """提取f980共振态配置"""
        if 'resonances' not in self.config or 'f980' not in self.config['resonances']:
            raise ValueError("配置文件中未找到f980共振态配置")
        
        f980_config = self.config['resonances']['f980']
        fixed_params = self.config.get('fixed_parameters', {})
        
        return {
            'name': f980_config['name'],
            'type': f980_config['type'],
            'mod_name': f980_config['mod_name'],
            'amplitude_wave': f980_config['amplitude_wave'],
            'propagators': f980_config['propagators'],
            'parameters': f980_config['parameters'],
            'coefficients': f980_config['coefficients'],
            'code_generation': f980_config['code_generation'],
            'fixed_parameters': fixed_params
        }
    
    def generate_calculate_BW_flatte980(self, target_file: str = "agent/fit_hvp_templates.py") -> str:
        """
        基于TOML配置生成calculate_BW_flatte980函数
        
        Args:
            target_file: 目标文件路径，用于分析现有结构
            
        Returns:
            生成的函数代码字符串
        """
        f980_config = self.extract_f980_config()
        
        # 生成函数代码
        function_code = f'''def calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg,
                          kk_f980_const, kk_f980_theta, phi_kk, f_kk, phif0_kk):
    """
    计算 BW×flatte980 贡献
    
    基于配置文件: {self.config_path}
    共振态: {f980_config['name']} ({f980_config['type']})
    模式标识: {f980_config['mod_name']}
    振幅波函数: {f980_config['amplitude_wave']}
    
    物理参数:
    - 质量: {f980_config['parameters']['mass']['value']:.6f} ± {f980_config['parameters']['mass'].get('error', 0.0):.6f}
    - g_kk: {f980_config['parameters']['g_kk']['value']:.6f} ± {f980_config['parameters']['g_kk'].get('error', 0.0):.6f}
    - rg: {f980_config['parameters']['rg']['value']:.6f} ± {f980_config['parameters']['rg'].get('error', 0.0):.6f}
    
    复数系数:
    - const1: {f980_config['coefficients']['const1']['value']:.6f} (固定)
    - const2: {f980_config['coefficients']['const2']['value']:.6f} ± {f980_config['coefficients']['const2'].get('error', 0.0):.6f}
    - theta1: {f980_config['coefficients']['theta1']['value']:.6f} (固定)  
    - theta2: {f980_config['coefficients']['theta2']['value']:.6f} ± {f980_config['coefficients']['theta2'].get('error', 0.0):.6f}
    
    Args:
        phi_mass: Phi介子质量 ({f980_config['fixed_parameters'].get('phi_mass', 1.02)} GeV)
        phi_width: Phi介子宽度 ({f980_config['fixed_parameters'].get('phi_width', 0.004)} GeV)
        kk_f980_mass: f980质量参数
        kk_f980_g_kk: f980-KK耦合常数
        kk_f980_rg: f980比值参数
        kk_f980_const: f980复数系数常数部分 (shape: [-1, 2])
        kk_f980_theta: f980复数系数相位部分 (shape: [-1, 2])
        phi_kk: Phi不变质量数据
        f_kk: f不变质量数据
        phif0_kk: Phi-f0振幅数据
        
    Returns:
        jax.numpy.array: 计算得到的振幅
    """
    # === Phi共振态传播子 ({f980_config['propagators']['phi_propagator']}) ===
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # === f980共振态传播子 ({f980_config['propagators']['f_propagator']}) ===
    # 使用Flatte形状，包含KK和ππ通道
    bw_f980 = np.moveaxis(
        vmap(partial(flatte980, Sbc=f_kk))(kk_f980_mass, kk_f980_g_kk, kk_f980_rg), 1, 0
    )
    
    # === 组合传播子 ===
    # Phi传播子与f980传播子的乘积
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f980)
    
    # === 复数系数构造 ===
    # 从常数和相位构造复数系数
    # const1 = {f980_config['coefficients']['const1']['value']} (固定)
    # const2 = 参数化
    # theta1 = {f980_config['coefficients']['theta1']['value']} (固定)
    # theta2 = 参数化
    const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
    
    # === 最终振幅计算 ===
    # 将复数系数应用到振幅上
    phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
    # 应用传播子
    phif = dplex.deinsum("ljk,lj->jk", phif, bw_combined)
    
    return phif'''
        
        return function_code
    
    def generate_lasso_calculate_BW_flatte980(self) -> str:
        """生成lasso版本的calculate_BW_flatte980函数（保留l维度用于约束）"""
        f980_config = self.extract_f980_config()
        
        function_code = f'''def lasso_calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg,
                                kk_f980_const, kk_f980_theta, phi_kk, f_kk, phif0_kk):
    """
    计算 BW×flatte980 贡献 (lasso版本，保留l维度用于约束)
    
    基于配置文件: {self.config_path}
    共振态: {f980_config['name']} ({f980_config['type']})
    
    此版本保留l维度，用于计算分支比约束。
    与标准版本的区别在于最终的Einstein求和中保留l维度。
    
    Args:
        phi_mass, phi_width: Phi介子参数
        kk_f980_mass, kk_f980_g_kk, kk_f980_rg: f980物理参数
        kk_f980_const, kk_f980_theta: f980复数系数
        phi_kk, f_kk, phif0_kk: 数据数组
        
    Returns:
        jax.numpy.array: 保留l维度的振幅 (shape: [l, j, k])
    """
    # === Phi共振态传播子 ===
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # === f980共振态传播子 ===
    bw_f980 = np.moveaxis(
        vmap(partial(flatte980, Sbc=f_kk))(kk_f980_mass, kk_f980_g_kk, kk_f980_rg), 1, 0
    )
    
    # === 组合传播子 ===
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f980)
    
    # === 复数系数构造 ===
    const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
    
    # === 最终振幅计算 (保留l维度) ===
    phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
    # 注意：这里是ljk,lj->ljk，保留l维度用于约束计算
    phif = dplex.deinsum("ljk,lj->ljk", phif, bw_combined)
    
    return phif'''
        
        return function_code
    
    def generate_parameter_extraction_code(self) -> str:
        """生成参数提取代码片段"""
        f980_config = self.extract_f980_config()
        
        # 根据配置生成参数提取代码
        code = f'''# === f980共振态参数提取 ===
# 基于配置: {f980_config['name']} ({f980_config['type']})

# 固定参数 (来自fixed_parameters)
phi_mass = np.array([{f980_config['fixed_parameters'].get('phi_mass', 1.02)}])
phi_width = np.array([{f980_config['fixed_parameters'].get('phi_width', 0.004)}])

# f980物理参数 (来自parameters section)
kk_f980_mass = np.array([args[0]])    # 质量: {f980_config['parameters']['mass']['value']:.6f}
kk_f980_g_kk = np.array([args[1]])    # g_kk: {f980_config['parameters']['g_kk']['value']:.6f}
kk_f980_rg = np.array([args[2]])      # rg: {f980_config['parameters']['rg']['value']:.6f}

# f980复数系数 (来自coefficients section)  
# const1={f980_config['coefficients']['const1']['value']} (固定), const2=args[3] (浮动)
kk_f980_const = np.array([{f980_config['coefficients']['const1']['value']}, args[3]]).reshape(-1, 2)
# theta1={f980_config['coefficients']['theta1']['value']} (固定), theta2=args[4] (浮动)
kk_f980_theta = np.array([{f980_config['coefficients']['theta1']['value']}, args[4]]).reshape(-1, 2)'''
        
        return code
    
    def generate_complete_functions(self) -> Dict[str, str]:
        """生成完整的函数集合"""
        functions = {}
        
        # 生成主计算函数
        functions['calculate_BW_flatte980'] = self.generate_calculate_BW_flatte980()
        
        # 生成lasso版本
        functions['lasso_calculate_BW_flatte980'] = self.generate_lasso_calculate_BW_flatte980()
        
        # 生成参数提取代码
        functions['parameter_extraction'] = self.generate_parameter_extraction_code()
        
        return functions
    
    def update_target_file(self, target_file: str = "agent/fit_hvp_templates.py"):
        """
        更新目标文件中的函数
        
        Args:
            target_file: 要更新的目标文件路径
        """
        if not os.path.exists(target_file):
            raise ValueError(f"目标文件不存在: {target_file}")
        
        functions = self.generate_complete_functions()
        
        # 读取现有文件
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单的函数替换（实际应用中可能需要更复杂的AST解析）
        updated_content = content
        
        # 替换calculate_BW_flatte980函数
        if 'def calculate_BW_flatte980(' in content:
            # 这里可以实现更智能的函数替换逻辑
            print("发现现有的calculate_BW_flatte980函数，准备更新...")
        
        # 输出生成的函数到控制台（演示用）
        print("="*80)
        print("生成的函数代码:")
        print("="*80)
        
        for func_name, func_code in functions.items():
            print(f"\n### {func_name} ###\n")
            print(func_code)
            print("\n" + "="*80)
        
        return functions
    
    def save_generated_code(self, output_path: str = "agent/generated_resonance_functions.py"):
        """将生成的代码保存到文件"""
        functions = self.generate_complete_functions()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成完整的Python文件
        header = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成的共振态计算函数
基于配置文件: {self.config_path}
生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

警告：此文件由 generate_resonance_functions.py 自动生成
手动修改可能会被覆盖
"""

import jax.numpy as np
from jax import vmap
from functools import partial
from dlib import dplex

# 需要导入的基础函数
# from fit_hvp_templates import BW, flatte980

'''
        
        # 组合所有函数
        full_code = header
        for func_name, func_code in functions.items():
            if func_name == 'parameter_extraction':
                full_code += f"\n\n# {func_name.upper()}\n"
                full_code += func_code
            else:
                full_code += f"\n\n{func_code}\n"
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        print(f"生成的代码已保存到: {output_path}")
        return output_path


def main():
    """主函数"""
    print("🚀 PWA共振态函数生成器启动!")
    print("📖 读取配置文件: agent/resonances_config.toml")
    
    # 创建生成器
    try:
        generator = ResonanceCodeGenerator()
        print("✅ 配置文件加载成功!")
        
        # 分析f980配置
        f980_config = generator.extract_f980_config()
        print(f"🔍 找到f980共振态配置:")
        print(f"   - 名称: {f980_config['name']}")
        print(f"   - 类型: {f980_config['type']}")
        print(f"   - 模式: {f980_config['mod_name']}")
        print(f"   - 振幅波: {f980_config['amplitude_wave']}")
        
        # 生成函数
        print("\n🛠️  正在生成函数...")
        functions = generator.update_target_file()
        
        # 保存到文件
        print("\n💾 保存生成的代码...")
        output_file = generator.save_generated_code()
        
        print(f"\n✨ 任务完成!")
        print(f"📁 输出文件: {output_file}")
        print(f"📊 生成函数数量: {len(functions)}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()