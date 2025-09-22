#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template file with structured sections for programmatic parsing.

Section identifiers for automatic parsing:
- COMMON_UTILITIES: Common utility functions and imports
- PATH_CONFIG: Path configuration and setup
- LOGGING_CONFIG: Logging configuration functions
- PHYSICS_FUNCTIONS: Physics calculation functions (resonance shapes)
- DATA_LOADING: Data loading function templates
- RESONANCE_CALCULATIONS: Composite resonance calculation templates
"""


# ==============================================================================
# SECTION: COMMON_UTILITIES
# Common utility functions
# ==============================================================================
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


# ==============================================================================
# SECTION: PATH_CONFIG
# Path configuration
# ==============================================================================
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from dlib import dplex

# ==============================================================================
# SECTION: LOGGING_CONFIG
# Logging configuration
# ==============================================================================
def setup_logging():
    """Setup logging configuration"""
    with open("config/logconfig_fit.json", "r") as config_file:
        LOGGING_CONFIG = json.load(config_file)
        logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("fit")

# ==============================================================================
# SECTION: PHYSICS_FUNCTIONS
# Physics calculation functions (resonance shape functions)
# ==============================================================================
def BW(m_,w_,Sbc):
    l = (Sbc.shape)[0]
    temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
    return dplex.ddivide(1.0, temp)

def BW_relativity(m_,w_,Sbc):
    gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
    k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
    l = (Sbc.shape)[0]
    temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
    return dplex.ddivide(k, temp)

def flatte980(m_,g_pipi,rg,Sbc):
    g_kk = rg * g_pipi
    m_k = 0.493677
    m_pi = 0.13957061
    rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
    rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
    tmp_A = dplex.dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
    return dplex.ddivide(1.0, tmp_A)

def flatte1270(m_,w_,Sbc):
    rm = m_ * m_
    gr = m_ * w_
    q2r = 0.25 * rm - 0.0194792
    b2r = q2r * (q2r + 0.1825) + 0.033306
    g11270 = gr * b2r / np.power(q2r,2.5)
    q2 = 0.25 * Sbc - 0.0194792
    b2 = q2 * (q2 + 0.1825) + 0.033306
    g1 = g11270 * np.power(q2,2.5) / b2
    tmp = dplex.dconstruct(Sbc - rm, g1)
    return dplex.ddivide(gr, tmp)

def flatte500(m_,b1,b2,b3,b4,b5,Sbc):
    m2 = m_*m_
    rp = 0.139556995
    mpi2d2 = 0.009739946882
    cro1 = np.sqrt(np.abs((Sbc-(2*rp)**2)*Sbc))/Sbc
    cro2 = np.sqrt(np.abs((m2-(2*rp)**2)*m2))/m2
    pip1 = np.sqrt(np.abs(1.0 - 0.3116765584/Sbc))/(1.0+np.exp(9.8-3.5*Sbc))
    pip2 = np.sqrt(np.abs(1.0 - 0.3116765584/m2))/(1.0+np.exp(9.8-3.5*m2)) # ?
    cgam1 = m_*(b1+b2*Sbc)*(Sbc-mpi2d2)/(m2-mpi2d2)*np.exp(-(Sbc-m2)/b3)*cro1/cro2
    cgam2 = m_*b4*pip1/pip2
    tmp = dplex.dconstruct(m2-Sbc, -b5*(cgam1+cgam2))
    return dplex.ddivide(1.0, tmp)


#==============================================================================
# SECTION: calculate_functions
#==============================================================================
def calculate_BW_BW(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = np.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(B_mass, B_width), 1, 0
    )
    propagator_combined = dplex.deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex.dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex.deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex.deinsum("ljk,lj->jk", result, propagator_combined)
    return result

def component_BW_BW(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = np.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(B_mass, B_width), 1, 0
    )
    propagator_combined = dplex.deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex.dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex.deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex.deinsum("ljk,lj->ljk", result, propagator_combined)
    return result

def calculate_BW_flatte1270(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = flatte1270(B_mass, B_width, f_kk)
    propagator_combined = dplex.deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex.dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex.deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex.deinsum("ljk,lj->jk", result, propagator_combined)
    return result

def component_BW_flatte1270(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = flatte1270(B_mass, B_width, f_kk)
    propagator_combined = dplex.deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex.dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex.deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex.deinsum("ljk,lj->ljk", result, propagator_combined)
    return result

#==============================================================================
# SECTION: likelihood_functions
#==============================================================================
def data_likelihood_kk(args):
    params = extract_parameters(args)
    data_phif0_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], data_f_kk,
        data_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    data_phif2_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    data_phif2_kk_BW_flatte1270 = calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    component_data_phif0_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], truth_f_kk,
        truth_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    component_data_phif2_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    component_data_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    sum_frac = np.sum(dplex.dabs(
        np.einsum("mljk->mjk", component_data_phif0_kk_BW_BW) +
        np.einsum("mljk->mjk", component_data_phif2_kk_BW_BW) +
        np.einsum("mljk->mjk", component_data_phif2_kk_BW_flatte1270)
    ))
    frac_f0_BW = np.sum(np.einsum("ljk->l", dplex.dabs(component_data_phif0_kk_BW_BW)) / sum_frac)
    frac_f2_BW = np.sum(np.einsum("ljk->l", dplex.dabs(component_data_phif2_kk_BW_BW)) / sum_frac)
    frac_f2_flatte = np.sum(np.einsum("ljk->l", dplex.dabs(component_data_phif2_kk_BW_flatte1270)) / sum_frac)
    total_frac = frac_f0_BW + frac_f2_BW + frac_f2_flatte
    step_function = np.power(total_frac - 1.03, 2.0) * constraint_strength
    total_amplitude = data_phif0_kk_BW_BW
    total_amplitude = total_amplitude + data_phif2_kk_BW_BW
    total_amplitude = total_amplitude + data_phif2_kk_BW_flatte1270
    likelihood = -np.sum(np.log(np.sum(dplex.dabs(total_amplitude), axis=1))) + 10000.0 * step_function
    return likelihood

def mc_likelihood_kk(args):
    params = extract_parameters(args)
    total_mc = calculate_BW_flatte980(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'], params['phif0_kk_BW_flatte980_rg'], mc_f_kk,
        mc_phif0_kk, params['phif0_kk_BW_flatte980_amplitude_consts'], params['phif0_kk_BW_flatte980_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], mc_f_kk,
        mc_phif0_kk, params['phif0_kk_BW_BW_amplitude_consts'], params['phif0_kk_BW_BW_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], mc_f_kk,
        mc_phif2_kk, params['phif2_kk_BW_flatte1270_amplitude_consts'], params['phif2_kk_BW_flatte1270_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], mc_f_kk,
        mc_phif2_kk, params['phif2_kk_BW_BW_amplitude_consts'], params['phif2_kk_BW_BW_amplitude_thetas']
    )
    return np.mean(np.sum(dplex.dabs(total_mc), axis=1))


#==============================================================================
# SECTION: combined_likelihood_function
#==============================================================================

def combined_likelihood(args):
    """组合似然函数: data_likelihood + datasize * log(mc_likelihood)"""
    # 计算data似然
    data_lh = data_likelihood_kk(args)
    
    # 计算MC似然
    mc_lh = mc_likelihood_kk(args)
    
    # 组合似然函数（使用类成员变量data_size）
    combined_lh = data_lh + data_size * np.log(mc_lh)
    
    return combined_lh

def hvp_combined_likelihood(args, vector):
    """计算Hessian-vector product for combined likelihood"""
    return jvp(grad(combined_likelihood), [args], [vector])[1]

#==============================================================================
# SECTION: prepare_data_parameters
#==============================================================================
args_list = onp.array([
    0.9794812115574156, 0.10678616326827592, 8.570187550432664, 
    0.1065182971468388, 0.03807025376236671, 1.6761965590304995,
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

#==============================================================================
# SECTION: load_data_section
#==============================================================================

data = load_data()
data = normalize_data(data)
jax_data = prepare_data_for_jax(data)

# 实验数据
data_phi_kk = jax_data['data_phi_kk']
data_f_kk = jax_data['data_f_kk']
data_phif0_kk = jax_data['data_phif0_kk']
data_phif2_kk = jax_data['data_phif2_kk']

# MC数据
mc_phi_kk = jax_data['mc_phi_kk']
mc_f_kk = jax_data['mc_f_kk']
mc_phif0_kk = jax_data['mc_phif0_kk']
mc_phif2_kk = jax_data['mc_phif2_kk']

# Truth数据（用于约束）
truth_phi_kk = jax_data['truth_phi_kk']
truth_f_kk = jax_data['truth_f_kk']
truth_phif0_kk = jax_data['truth_phif0_kk']
truth_phif2_kk = jax_data['truth_phif2_kk']

# 权重数据
wt_data_kk = jax_data['wt_data_kk']

# 数据大小（用于似然函数计算）
data_size = len(data_phi_kk)

#==============================================================================
# SECTION: main_section
#==============================================================================

if __name__  == "__main__":
    """使用Newton-CG方法和HVP的拟合函数"""
    logger = setup_logging()
    logger.info("开始HVP优化版PWA拟合（Newton-CG方法）")
    
    # 设置JAX
    config.update("jax_enable_x64", True)
    
    # 约束强度（可调节）
    constraint_strength = 0.0
    
    # 编译JAX函数（使用HVP版本）
    logger.info("编译JAX函数（HVP版本）...")
    jit_likelihood = jit(combined_likelihood)
    jit_grad = jit(grad(combined_likelihood))
    jit_hvp = jit(hvp_combined_likelihood)
    
    # 测试编译
    test_result = jit_likelihood(args_list)
    logger.info(f"初始似然值: {test_result}")
    
    # 测试HVP编译
    test_vector = onp.ones_like(args_list)
    test_hvp = jit_hvp(args_list, test_vector)
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
        x0=args_list,
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