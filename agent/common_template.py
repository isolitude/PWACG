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
# SECTION: likelihood_functions
# likelihood function templates
#==============================================================================
def data_likelihood_kk(args):
    params = extract_parameters(args)
    data_phif0_kk_BW_BW = calculate_BW_BW(
        phi_mass, phi_width, data_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], data_f_kk,
        data_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    data_phif2_kk_BW_BW = calculate_BW_BW(
        phi_mass, phi_width, data_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    data_phif2_kk_BW_flatte1270 = calculate_BW_flatte1270(
        phi_mass, phi_width, data_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    component_data_phif0_kk_BW_BW = component_BW_BW(
        phi_mass, phi_width, truth_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], truth_f_kk,
        truth_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    component_data_phif2_kk_BW_BW = component_BW_BW(
        phi_mass, phi_width, truth_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    component_data_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        phi_mass, phi_width, truth_phi_kk,
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
        phi_mass, phi_width, mc_phi_kk,
        params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'], params['phif0_kk_BW_flatte980_rg'], mc_f_kk,
        mc_phif0_kk, params['phif0_kk_BW_flatte980_amplitude_consts'], params['phif0_kk_BW_flatte980_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_BW(
        phi_mass, phi_width, mc_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], mc_f_kk,
        mc_phif0_kk, params['phif0_kk_BW_BW_amplitude_consts'], params['phif0_kk_BW_BW_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_flatte1270(
        phi_mass, phi_width, mc_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], mc_f_kk,
        mc_phif2_kk, params['phif2_kk_BW_flatte1270_amplitude_consts'], params['phif2_kk_BW_flatte1270_amplitude_thetas']
    )
    total_mc = total_mc + calculate_BW_BW(
        phi_mass, phi_width, mc_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], mc_f_kk,
        mc_phif2_kk, params['phif2_kk_BW_BW_amplitude_consts'], params['phif2_kk_BW_BW_amplitude_thetas']
    )
    return np.mean(np.sum(dplex.dabs(total_mc), axis=1))


#==============================================================================
# SECTION: combined_likelihood_function
# likelihood function templates
#==============================================================================

def combined_likelihood(self, args):
    """组合似然函数: data_likelihood + datasize * log(mc_likelihood)"""
    # 计算data似然
    data_lh = data_likelihood_kk(args)
    
    # 计算MC似然
    mc_lh = mc_likelihood_kk(args)
    
    # 组合似然函数（使用类成员变量data_size）
    combined_lh = data_lh + data_size * np.log(mc_lh)
    
    return combined_lh

def hvp_combined_likelihood(self, args, vector):
    """计算Hessian-vector product for combined likelihood"""
    return jvp(grad(combined_likelihood), [args], [vector])[1]

def set_constraint_strength(self, strength):
    """设置约束强度"""
    constraint_strength = strength