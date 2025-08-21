#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成的共振态计算函数
基于配置文件: agent/resonances_config.toml
生成时间: 2025-08-21 23:40:18

警告：此文件由 generate_resonance_functions.py 自动生成
手动修改可能会被覆盖
"""

import jax.numpy as np
from jax import vmap
from functools import partial
from dlib import dplex

# 需要导入的基础函数
# from fit_hvp_templates import BW, flatte980



def calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg,
                          kk_f980_const, kk_f980_theta, phi_kk, f_kk, phif0_kk):
    """
    计算 BW×flatte980 贡献
    
    基于配置文件: agent/resonances_config.toml
    共振态: f980 (flatte)
    模式标识: phif0_980_kk
    振幅波函数: phif0_kk
    
    物理参数:
    - 质量: 0.979481 ± 0.002911
    - g_kk: 0.106786 ± 0.010261
    - rg: 8.570188 ± 1.348490
    
    复数系数:
    - const1: 0.100000 (固定)
    - const2: 0.106518 ± 0.003833
    - theta1: 0.100000 (固定)  
    - theta2: 0.038070 ± 0.003238
    
    Args:
        phi_mass: Phi介子质量 (1.02 GeV)
        phi_width: Phi介子宽度 (0.004 GeV)
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
    # === Phi共振态传播子 (BW) ===
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # === f980共振态传播子 (flatte980) ===
    # 使用Flatte形状，包含KK和ππ通道
    bw_f980 = np.moveaxis(
        vmap(partial(flatte980, Sbc=f_kk))(kk_f980_mass, kk_f980_g_kk, kk_f980_rg), 1, 0
    )
    
    # === 组合传播子 ===
    # Phi传播子与f980传播子的乘积
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f980)
    
    # === 复数系数构造 ===
    # 从常数和相位构造复数系数
    # const1 = 0.1 (固定)
    # const2 = 参数化
    # theta1 = 0.1 (固定)
    # theta2 = 参数化
    const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
    
    # === 最终振幅计算 ===
    # 将复数系数应用到振幅上
    phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
    # 应用传播子
    phif = dplex.deinsum("ljk,lj->jk", phif, bw_combined)
    
    return phif


def lasso_calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg,
                                kk_f980_const, kk_f980_theta, phi_kk, f_kk, phif0_kk):
    """
    计算 BW×flatte980 贡献 (lasso版本，保留l维度用于约束)
    
    基于配置文件: agent/resonances_config.toml
    共振态: f980 (flatte)
    
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
    
    return phif


# PARAMETER_EXTRACTION
# === f980共振态参数提取 ===
# 基于配置: f980 (flatte)

# 固定参数 (来自fixed_parameters)
phi_mass = np.array([1.02])
phi_width = np.array([0.004])

# f980物理参数 (来自parameters section)
kk_f980_mass = np.array([args[0]])    # 质量: 0.979481
kk_f980_g_kk = np.array([args[1]])    # g_kk: 0.106786
kk_f980_rg = np.array([args[2]])      # rg: 8.570188

# f980复数系数 (来自coefficients section)  
# const1=0.1 (固定), const2=args[3] (浮动)
kk_f980_const = np.array([0.1, args[3]]).reshape(-1, 2)
# theta1=0.1 (固定), theta2=args[4] (浮动)
kk_f980_theta = np.array([0.1, args[4]]).reshape(-1, 2)