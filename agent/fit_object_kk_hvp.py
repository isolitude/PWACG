#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 PWA 拟合脚本 - 使用Newton-CG方法和HVP优化
基于fit_object_kk_sample.py，添加了Hessian-vector product支持
"""

import copy
import json
import logging
import logging.config
import os
import time
import numpy as onp
from functools import partial
from scipy.optimize import minimize

import jax.numpy as np
from jax import device_put, grad, jit, vmap, jvp
from jax import config

import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from dlib import dplex

# ==============================================================================
# 日志配置
# ==============================================================================
def setup_logging():
    """设置日志配置"""
    with open("config/logconfig_fit.json", "r") as config_file:
        LOGGING_CONFIG = json.load(config_file)
        logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("fit")

# ==============================================================================
# 数据加载函数
# ==============================================================================
def load_data():
    """加载所有需要的数据"""
    data = {}
    
    # 加载实际数据
    data['data_phif0_kk'] = onp.load("data/real_data/phif0_kk.npy")
    data['data_phif2_kk'] = onp.load("data/real_data/phif2_kk.npy")
    data['data_phi_kk'] = onp.load("data/real_data/phi_kk.npy")
    data['data_f_kk'] = onp.load("data/real_data/f_kk.npy")
    
    # 加载MC数据
    data['mc_phif0_kk'] = onp.load("data/mc_truth/phif0_kk.npy")
    data['mc_phif2_kk'] = onp.load("data/mc_truth/phif2_kk.npy")
    data['mc_phi_kk'] = onp.load("data/mc_truth/phi_kk.npy")
    data['mc_f_kk'] = onp.load("data/mc_truth/f_kk.npy")
    
    # MC真值数据（用于约束）
    data['truth_phif0_kk'] = data['mc_phif0_kk'][:, 0:150000]
    data['truth_phif2_kk'] = data['mc_phif2_kk'][:, 0:150000]
    data['truth_phi_kk'] = data['mc_phi_kk'][0:150000]
    data['truth_f_kk'] = data['mc_f_kk'][0:150000]
    
    # 权重数据
    try:
        data['wt_data_kk'] = onp.load("data/weight/weight_kk.npy")
    except FileNotFoundError:
        data['wt_data_kk'] = onp.ones_like(data['data_phi_kk'])
    
    return data

def normalize_data(data):
    """数据归一化处理"""
    # 计算归一化因子
    regular_phif0_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_phif0_kk'])**2, axis=2)), axis=1
    )
    regular_phif2_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_phif2_kk'])**2, axis=2)), axis=1
    )
    
    # 应用归一化
    data['data_phif0_kk'] = onp.einsum("jkl,j->jkl", data['data_phif0_kk'], regular_phif0_kk)
    data['data_phif2_kk'] = onp.einsum("jkl,j->jkl", data['data_phif2_kk'], regular_phif2_kk)
    data['mc_phif0_kk'] = onp.einsum("jkl,j->jkl", data['mc_phif0_kk'], regular_phif0_kk)
    data['mc_phif2_kk'] = onp.einsum("jkl,j->jkl", data['mc_phif2_kk'], regular_phif2_kk)
    data['truth_phif0_kk'] = onp.einsum("jkl,j->jkl", data['truth_phif0_kk'], regular_phif0_kk)
    data['truth_phif2_kk'] = onp.einsum("jkl,j->jkl", data['truth_phif2_kk'], regular_phif2_kk)
    
    return data

def prepare_data_for_jax(data, device=None):
    """将数据转换为JAX格式并放到指定设备"""
    jax_data = {}
    for key, value in data.items():
        jax_data[key] = device_put(np.array(value), device=device)
    return jax_data

# ==============================================================================
# 物理计算函数（共振态形状函数）
# ==============================================================================
def BW(m_, w_, Sbc):
    """Breit-Wigner 共振态形状"""
    l = (Sbc.shape)[0]
    temp = dplex.dconstruct(m_*m_ - Sbc, -m_*w_*np.ones(l))
    return dplex.ddivide(1.0, temp)

def flatte980(m_, g_pipi, rg, Sbc):
    """Flatte 共振态形状 (f980)"""
    g_kk = rg * g_pipi
    m_k = 0.493677
    m_pi = 0.13957061
    rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
    rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
    tmp_A = dplex.dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
    return dplex.ddivide(1.0, tmp_A)

def flatte1270(m_, w_, Sbc):
    """Flatte 共振态形状 (f1270)"""
    rm = m_ * m_
    gr = m_ * w_
    q2r = 0.25 * rm - 0.0194792
    b2r = q2r * (q2r + 0.1825) + 0.033306
    g11270 = gr * b2r / np.power(q2r, 2.5)
    q2 = 0.25 * Sbc - 0.0194792
    b2 = q2 * (q2 + 0.1825) + 0.033306
    g1 = g11270 * np.power(q2, 2.5) / b2
    tmp = dplex.dconstruct(Sbc - rm, g1)
    return dplex.ddivide(gr, tmp)

# ==============================================================================
# 复合共振态计算函数
# ==============================================================================
def calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg,
                          kk_f980_const, kk_f980_theta, phi_kk, f_kk, phif0_kk):
    """计算 BW×flatte980 贡献"""
    # Phi 共振态
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # f980 共振态
    bw_f980 = np.moveaxis(
        vmap(partial(flatte980, Sbc=f_kk))(kk_f980_mass, kk_f980_g_kk, kk_f980_rg), 1, 0
    )
    
    # 组合传播子
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f980)
    
    # 复数系数
    const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
    
    # 最终幅度
    phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
    phif = dplex.deinsum("ljk,lj->jk", phif, bw_combined)
    
    return phif

def calculate_BW_BW(phi_mass, phi_width, kk_f0_mass, kk_f0_width,
                   kk_f0_const, kk_f0_theta, phi_kk, f_kk, phif_kk):
    """计算 BW×BW 贡献"""
    # Phi 共振态
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # f0 共振态
    bw_f0 = np.moveaxis(vmap(partial(BW, Sbc=f_kk))(kk_f0_mass, kk_f0_width), 1, 0)
    
    # 组合传播子
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f0)
    
    # 复数系数
    const_ph = dplex.dconstruct(kk_f0_const, kk_f0_theta)
    
    # 最终幅度
    phif = dplex.deinsum_ord("ijk,li->ljk", phif_kk, const_ph)
    phif = dplex.deinsum("ljk,lj->jk", phif, bw_combined)
    
    return phif

def calculate_BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width,
                           kk_f1270_const, kk_f1270_theta, phi_kk, f_kk, phif2_kk):
    """计算 BW×flatte1270 贡献"""
    # Phi 共振态
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # f1270 共振态
    bw_f1270 = np.moveaxis(
        vmap(partial(flatte1270, Sbc=f_kk))(kk_f1270_mass, kk_f1270_width), 1, 0
    )
    
    # 组合传播子
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f1270)
    
    # 复数系数
    const_ph = dplex.dconstruct(kk_f1270_const, kk_f1270_theta)
    
    # 最终幅度
    phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk, const_ph)
    phif = dplex.deinsum("ljk,lj->jk", phif, bw_combined)
    
    return phif

# ==============================================================================
# Lasso版本的计算函数（保留l维度用于约束）
# ==============================================================================
def lasso_calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg,
                                kk_f980_const, kk_f980_theta, phi_kk, f_kk, phif0_kk):
    """计算 BW×flatte980 贡献 (lasso版本，保留l维度)"""
    # Phi 共振态
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # f980 共振态
    bw_f980 = np.moveaxis(
        vmap(partial(flatte980, Sbc=f_kk))(kk_f980_mass, kk_f980_g_kk, kk_f980_rg), 1, 0
    )
    
    # 组合传播子
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f980)
    
    # 复数系数
    const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
    
    # 最终幅度 (注意：这里是ljk,lj->ljk，保留l维度)
    phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
    phif = dplex.deinsum("ljk,lj->ljk", phif, bw_combined)  # 保留l维度
    
    return phif

def lasso_calculate_BW_BW(phi_mass, phi_width, kk_f0_mass, kk_f0_width,
                         kk_f0_const, kk_f0_theta, phi_kk, f_kk, phif_kk):
    """计算 BW×BW 贡献 (lasso版本，保留l维度)"""
    # Phi 共振态
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # f0 共振态
    bw_f0 = np.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(kk_f0_mass, kk_f0_width), 1, 0
    )
    
    # 组合传播子
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f0)
    
    # 复数系数
    const_ph = dplex.dconstruct(kk_f0_const, kk_f0_theta)
    
    # 最终幅度 (注意：这里是ljk,lj->ljk，保留l维度)
    phif = dplex.deinsum_ord("ijk,li->ljk", phif_kk, const_ph)
    phif = dplex.deinsum("ljk,lj->ljk", phif, bw_combined)  # 保留l维度
    
    return phif

def lasso_calculate_BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width,
                                 kk_f1270_const, kk_f1270_theta, phi_kk, f_kk, phif2_kk):
    """计算 BW×flatte1270 贡献 (lasso版本，保留l维度)"""
    # Phi 共振态
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # f1270 共振态
    bw_f1270 = np.moveaxis(
        vmap(partial(flatte1270, Sbc=f_kk))(kk_f1270_mass, kk_f1270_width), 1, 0
    )
    
    # 组合传播子
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f1270)
    
    # 复数系数
    const_ph = dplex.dconstruct(kk_f1270_const, kk_f1270_theta)
    
    # 最终幅度 (注意：这里是ljk,lj->ljk，保留l维度)
    phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk, const_ph)
    phif = dplex.deinsum("ljk,lj->ljk", phif, bw_combined)  # 保留l维度
    
    return phif

# ==============================================================================
# 参数提取工具函数
# ==============================================================================
def extract_parameters(args):
    """提取所有物理参数的公共函数"""
    return {
        'phi_mass': np.array([1.02]),
        'phi_width': np.array([0.004]),
        'kk_f980_mass': np.array([args[0]]),
        'kk_f980_g_kk': np.array([args[1]]),
        'kk_f980_rg': np.array([args[2]]),
        'kk_f980_const': np.array([0.1, args[3]]).reshape(-1, 2),
        'kk_f980_theta': np.array([0.1, args[4]]).reshape(-1, 2),
        'kk_f0_mass': np.array([args[5]]),
        'kk_f0_width': np.array([args[6]]),
        'kk_f0_const': np.array([args[7], args[8]]).reshape(-1, 2),
        'kk_f0_theta': np.array([args[9], args[10]]).reshape(-1, 2),
        'kk_f1270_mass': np.array([args[11]]),
        'kk_f1270_width': np.array([args[12]]),
        'kk_f1270_const': np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5),
        'kk_f1270_theta': np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5),
        'kk_f2_mass': np.array([args[23], args[24], args[25]]),
        'kk_f2_width': np.array([args[26], args[27], args[28]]),
        'kk_f2_const': np.array([args[29], args[30], args[31], args[32], args[33], 
                               args[34], args[35], args[36], args[37], args[38], 
                               args[39], args[40], args[41], args[42], args[43]]).reshape(-1, 5),
        'kk_f2_theta': np.array([args[44], args[45], args[46], args[47], args[48], 
                               args[49], args[50], args[51], args[52], args[53], 
                               args[54], args[55], args[56], args[57], args[58]]).reshape(-1, 5)
    }

# ==============================================================================
# 似然计算类（最优化版本，带HVP支持）
# ==============================================================================
class PWALikelihoodCalculatorHVP:
    """PWA似然计算类 - 数据预解包，支持HVP优化"""
    
    def __init__(self, jax_data):
        """初始化时预解包所有数据，避免重复访问"""
        # 实际数据
        self.data_phi_kk = jax_data['data_phi_kk']
        self.data_f_kk = jax_data['data_f_kk']
        self.data_phif0_kk = jax_data['data_phif0_kk']
        self.data_phif2_kk = jax_data['data_phif2_kk']
        
        # MC数据
        self.mc_phi_kk = jax_data['mc_phi_kk']
        self.mc_f_kk = jax_data['mc_f_kk']
        self.mc_phif0_kk = jax_data['mc_phif0_kk']
        self.mc_phif2_kk = jax_data['mc_phif2_kk']
        
        # Truth数据（用于约束）
        self.truth_phi_kk = jax_data['truth_phi_kk']
        self.truth_f_kk = jax_data['truth_f_kk']
        self.truth_phif0_kk = jax_data['truth_phif0_kk']
        self.truth_phif2_kk = jax_data['truth_phif2_kk']
        
        # 权重数据
        self.wt_data_kk = jax_data['wt_data_kk']
        
        # 数据大小（用于似然函数计算）
        self.data_size = len(self.data_phi_kk)
        
        # 约束强度（可调节）
        self.constraint_strength = 0.0
    
    def data_likelihood_kk(self, args):
        """数据似然函数（类成员版本，最高效）"""
        # 提取参数（使用公共函数）
        params = extract_parameters(args)
        
        # 计算各个贡献（直接使用成员变量）
        data_phif0_kk_BW_flatte980 = calculate_BW_flatte980(
            params['phi_mass'], params['phi_width'], params['kk_f980_mass'], 
            params['kk_f980_g_kk'], params['kk_f980_rg'],
            params['kk_f980_const'], params['kk_f980_theta'], 
            self.data_phi_kk, self.data_f_kk, self.data_phif0_kk
        )
        
        data_phif0_kk_BW_BW = calculate_BW_BW(
            params['phi_mass'], params['phi_width'], params['kk_f0_mass'], params['kk_f0_width'],
            params['kk_f0_const'], params['kk_f0_theta'],
            self.data_phi_kk, self.data_f_kk, self.data_phif0_kk
        )
        
        data_phif2_kk_BW_flatte1270 = calculate_BW_flatte1270(
            params['phi_mass'], params['phi_width'], params['kk_f1270_mass'], params['kk_f1270_width'],
            params['kk_f1270_const'], params['kk_f1270_theta'],
            self.data_phi_kk, self.data_f_kk, self.data_phif2_kk
        )
        
        data_phif2_kk_BW_BW = calculate_BW_BW(
            params['phi_mass'], params['phi_width'], params['kk_f2_mass'], params['kk_f2_width'],
            params['kk_f2_const'], params['kk_f2_theta'],
            self.data_phi_kk, self.data_f_kk, self.data_phif2_kk
        )
        
        # ========== 完整的约束项计算 ==========
        # 使用lasso版本的函数计算所有约束项
        lasso_data_phif0_kk_BW_flatte980 = lasso_calculate_BW_flatte980(
            params['phi_mass'], params['phi_width'], params['kk_f980_mass'], 
            params['kk_f980_g_kk'], params['kk_f980_rg'],
            params['kk_f980_const'], params['kk_f980_theta'],
            self.truth_phi_kk, self.truth_f_kk, self.truth_phif0_kk
        )
        
        lasso_data_phif0_kk_BW_BW = lasso_calculate_BW_BW(
            params['phi_mass'], params['phi_width'], params['kk_f0_mass'], params['kk_f0_width'],
            params['kk_f0_const'], params['kk_f0_theta'],
            self.truth_phi_kk, self.truth_f_kk, self.truth_phif0_kk
        )
        
        lasso_data_phif2_kk_BW_flatte1270 = lasso_calculate_BW_flatte1270(
            params['phi_mass'], params['phi_width'], params['kk_f1270_mass'], params['kk_f1270_width'],
            params['kk_f1270_const'], params['kk_f1270_theta'],
            self.truth_phi_kk, self.truth_f_kk, self.truth_phif2_kk
        )
        
        lasso_data_phif2_kk_BW_BW = lasso_calculate_BW_BW(
            params['phi_mass'], params['phi_width'], params['kk_f2_mass'], params['kk_f2_width'],
            params['kk_f2_const'], params['kk_f2_theta'],
            self.truth_phi_kk, self.truth_f_kk, self.truth_phif2_kk
        )
        
        # 计算总的约束分母（所有共振态的总和）
        sum_frac = np.sum(dplex.dabs(
            np.einsum("mljk->mjk", lasso_data_phif0_kk_BW_flatte980) +
            np.einsum("mljk->mjk", lasso_data_phif0_kk_BW_BW) +
            np.einsum("mljk->mjk", lasso_data_phif2_kk_BW_flatte1270) +
            np.einsum("mljk->mjk", lasso_data_phif2_kk_BW_BW)
        ))
        
        # 计算各个共振态的分数约束
        frac_f980_flatte = np.sum(np.einsum("ljk->l", dplex.dabs(lasso_data_phif0_kk_BW_flatte980)) / sum_frac)
        frac_f0_BW = np.sum(np.einsum("ljk->l", dplex.dabs(lasso_data_phif0_kk_BW_BW)) / sum_frac)
        frac_f1270_flatte = np.sum(np.einsum("ljk->l", dplex.dabs(lasso_data_phif2_kk_BW_flatte1270)) / sum_frac)
        frac_f2_BW = np.sum(np.einsum("ljk->l", dplex.dabs(lasso_data_phif2_kk_BW_BW)) / sum_frac)
        
        # 总分数约束（应该接近目标值，如1.03）
        total_frac = frac_f980_flatte + frac_f0_BW + frac_f1270_flatte + frac_f2_BW
        
        # 约束函数（可调节约束强度）
        step_function = np.power(total_frac - 1.03, 2.0) * self.constraint_strength
        
        # 总似然（使用逐步累加减少临时变量）
        total_amplitude = data_phif0_kk_BW_flatte980
        total_amplitude = total_amplitude + data_phif0_kk_BW_BW
        total_amplitude = total_amplitude + data_phif2_kk_BW_flatte1270
        total_amplitude = total_amplitude + data_phif2_kk_BW_BW
        
        likelihood = -np.sum(np.log(np.sum(dplex.dabs(total_amplitude), axis=1))) + 10000.0 * step_function
        
        return likelihood
    
    def mc_likelihood_kk(self, args):
        """MC似然函数（类成员版本，最高效）"""
        # 复用参数提取函数
        params = extract_parameters(args)
        
        # 计算所有MC贡献（逐步累加，直接使用成员变量）
        total_mc = calculate_BW_flatte980(
            params['phi_mass'], params['phi_width'], params['kk_f980_mass'], 
            params['kk_f980_g_kk'], params['kk_f980_rg'],
            params['kk_f980_const'], params['kk_f980_theta'],
            self.mc_phi_kk, self.mc_f_kk, self.mc_phif0_kk
        )
        
        total_mc = total_mc + calculate_BW_BW(
            params['phi_mass'], params['phi_width'], params['kk_f0_mass'], params['kk_f0_width'],
            params['kk_f0_const'], params['kk_f0_theta'],
            self.mc_phi_kk, self.mc_f_kk, self.mc_phif0_kk
        )
        
        total_mc = total_mc + calculate_BW_flatte1270(
            params['phi_mass'], params['phi_width'], params['kk_f1270_mass'], params['kk_f1270_width'],
            params['kk_f1270_const'], params['kk_f1270_theta'],
            self.mc_phi_kk, self.mc_f_kk, self.mc_phif2_kk
        )
        
        total_mc = total_mc + calculate_BW_BW(
            params['phi_mass'], params['phi_width'], params['kk_f2_mass'], params['kk_f2_width'],
            params['kk_f2_const'], params['kk_f2_theta'],
            self.mc_phi_kk, self.mc_f_kk, self.mc_phif2_kk
        )
        
        # 返回总MC积分
        return np.mean(np.sum(dplex.dabs(total_mc),axis=1))
    
    def combined_likelihood(self, args):
        """组合似然函数: data_likelihood + datasize * log(mc_likelihood)"""
        # 计算data似然
        data_lh = self.data_likelihood_kk(args)
        
        # 计算MC似然
        mc_lh = self.mc_likelihood_kk(args)
        
        # 组合似然函数（使用类成员变量data_size）
        combined_lh = data_lh + self.data_size * np.log(mc_lh)
        
        return combined_lh
    
    def hvp_combined_likelihood(self, args, vector):
        """计算Hessian-vector product for combined likelihood"""
        return jvp(grad(self.combined_likelihood), [args], [vector])[1]
    
    def set_constraint_strength(self, strength):
        """设置约束强度"""
        self.constraint_strength = strength

# ==============================================================================
# 主拟合函数（使用Newton-CG和HVP）
# ==============================================================================
def hvp_fit():
    """使用Newton-CG方法和HVP的拟合函数"""
    logger = setup_logging()
    logger.info("开始HVP优化版PWA拟合（Newton-CG方法）")
    
    # 设置JAX
    config.update("jax_enable_x64", True)
    
    # 加载和准备数据
    logger.info("加载数据...")
    data = load_data()
    data = normalize_data(data)
    jax_data = prepare_data_for_jax(data)
    
    # 创建HVP优化版似然计算器
    logger.info("创建HVP优化版似然计算器...")
    calculator = PWALikelihoodCalculatorHVP(jax_data)
    logger.info(f"数据大小: {calculator.data_size}")
    
    # 初始参数
    args_list = onp.array([
        1.02, 0.004, 0.9794812115574156, 0.10678616326827592, 8.570187550432664, 
        0.1, 0.1065182971468388, 0.1, 0.03807025376236671, 1.6761965590304995,
        0.16270071108440043, 0.02201375198386279, 0.007433204877151768, 
        0.008302300118288414, -0.018544178045626723, 1.2896149318644679,
        0.1959796900380022, -0.013533706721054747, -0.01650921272412254,
        0.015339403283337268, 0.029798824827026338, 0.020982478502465995,
        0.05458389002821978, 0.016032334798079934, 0.017179622009723918,
        -0.050143087437086675, -0.008718872479379924, 1.5222602842746435,
        2.1619576785269476, 2.547889297662712, 0.08576525399315078,
        0.15906049251159413, 0.324001266488012, -0.005345214125595505,
        0.0031770703345810553, -0.0036743677603351056, 0.004813366936315499,
        0.010000673114238258, 0.004643720949518616, -0.0009682564746806725,
        -0.0029804674908844213, 0.008315390493289363, 0.0005819695034846763,
        -0.15266341362240637, -0.05030214288962155, -0.01856511044577769,
        0.0908719088071242, 0.029544599934778714, -0.010529063356530807,
        0.003910028530572422, 0.0031787953173277725, 0.0333658928584713,
        -0.006053516058970728, 0.00545992748088964, -0.012498776031677225,
        0.002196563552197887, -0.016814307435916394, -0.007232946300343621,
        0.06277085590848677, -0.0031366601030189344, 0.13756083806604205,
        -0.0033564526519642953, 0.06849361819185686
    ])
    
    float_list = onp.array([2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 
                           33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
                           47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62])
    
    args_float = args_list[float_list]
    
    # 编译JAX函数（使用HVP版本）
    logger.info("编译JAX函数（HVP版本）...")
    jit_likelihood = jit(calculator.combined_likelihood)
    jit_grad = jit(grad(calculator.combined_likelihood))
    jit_hvp = jit(calculator.hvp_combined_likelihood)
    
    # 测试编译
    test_result = jit_likelihood(args_float)
    logger.info(f"初始似然值: {test_result}")
    
    # 测试HVP编译
    test_vector = onp.ones_like(args_float)
    test_hvp = jit_hvp(args_float, test_vector)
    logger.info(f"HVP测试完成，结果形状: {test_hvp.shape}")
    
    # 定义callback函数
    def my_callback(x):
        current_likelihood = jit_likelihood(x)
        logger.info(f"当前似然值: {current_likelihood}")
    
    # 定义HVP函数（适应scipy.optimize接口）
    def hessp(x, p):
        return onp.array(jit_hvp(x, p))
    
    # 优化（使用Newton-CG方法）
    logger.info("开始优化（Newton-CG + HVP）...")
    start_time = time.time()
    
    result = minimize(
        fun=lambda x: float(jit_likelihood(x)),
        x0=args_float,
        jac=lambda x: onp.array(jit_grad(x)),
        hessp=hessp,
        method="Newton-CG",
        callback=my_callback,
        options={"disp": False, "xtol": 1e-8}
    )
    
    end_time = time.time()
    
    # 结果报告
    logger.info("="*50)
    logger.info(f"HVP优化完成!")
    logger.info(f"成功: {result.success}")
    logger.info(f"最终似然值: {result.fun}")
    logger.info(f"迭代次数: {result.nit}")
    logger.info(f"函数调用次数: {result.nfev}")
    logger.info(f"梯度调用次数: {result.njev}")
    logger.info(f"Hessian调用次数: {result.nhev}")
    logger.info(f"优化时间: {end_time - start_time:.2f} 秒")
    logger.info(f"优化信息: {result.message}")
    logger.info("="*50)
    
    # 保存结果
    final_args = copy.deepcopy(args_list)
    for i, locate in enumerate(float_list):
        final_args[locate] = result.x[i]
    
    # 简单的结果保存
    results = {
        "success": bool(result.success),
        "final_likelihood": float(result.fun),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
        "hessian_evaluations": int(result.nhev),
        "optimization_time": float(end_time - start_time),
        "parameters": final_args.tolist(),
        "optimized_parameters": result.x.tolist(),
        "float_indices": float_list.tolist(),
        "optimization_method": "Newton-CG_with_HVP"
    }
    
    # 保存到JSON文件
    output_dir = "output/fit"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/hvp_fit_result.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"结果已保存到 {output_dir}/hvp_fit_result.json")
    
    return results

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    try:
        results = hvp_fit()
        print("HVP拟合成功完成!")
        print(f"最终似然值: {results['final_likelihood']}")
        print(f"Hessian调用次数: {results['hessian_evaluations']}")
    except Exception as e:
        print(f"拟合过程中出现错误: {e}")
        import traceback
        traceback.print_exc()