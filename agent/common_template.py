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
def BW(self, m_,w_,Sbc):
    l = (Sbc.shape)[0]
    temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
    return dplex.ddivide(1.0, temp)

def BW_relativity(self, m_,w_,Sbc):
    gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
    k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
    l = (Sbc.shape)[0]
    temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
    return dplex.ddivide(k, temp)

def flatte980(self,m_,g_pipi,rg,Sbc):
    g_kk = rg * g_pipi
    m_k = 0.493677
    m_pi = 0.13957061
    rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
    rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
    tmp_A = dplex.dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
    return dplex.ddivide(1.0, tmp_A)

def flatte1270(self,m_,w_,Sbc):
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

def flatte500(self,m_,b1,b2,b3,b4,b5,Sbc):
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


# ==============================================================================
# SECTION: RESONANCE_CALCULATIONS
# Composite resonance calculation function template
# ==============================================================================
def calculate_propagatorA_propagatorB(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg,
                          kk_f980_const, kk_f980_theta, phi_kk, f_kk, phif0_kk):
    """Calculate BW×flatte980 contribution"""
    # Phi resonance
    bw_phi = BW(phi_mass, phi_width, phi_kk)
    
    # f980 resonance
    bw_f980 = np.moveaxis(
        vmap(partial(flatte980, Sbc=f_kk))(kk_f980_mass, kk_f980_g_kk, kk_f980_rg), 1, 0
    )
    
    # Combined propagator
    bw_combined = dplex.deinsum("j, ij->ij", bw_phi, bw_f980)
    
    # Complex coefficient
    const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
    
    # Final amplitude
    phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
    phif = dplex.deinsum("ljk,lj->jk", phif, bw_combined)
    
    return phif


# ==============================================================================
# SECTION: DATA_LOADING
# Data loading function template
# ==============================================================================
def load_data():
    """Load all required data"""
    data = {}
    
    # Load real data
    data['data_{var}'] = onp.load("data/real_data/{var}.npy")

    # Load MC data
    data['mc_{var}'] = onp.load("data/mc_truth/{var}.npy")

    # MC truth data (for constraints)
    data['truth_{var}'] = data['mc_{var}'][:, 0:150000]
    
    # Weight data
    try:
        data['wt_data_kk'] = onp.load("data/weight/weight_kk.npy")
    except FileNotFoundError:
        data['wt_data_kk'] = onp.ones_like(data['data_phi_kk'])
    
    return data

def normalize_data(data):
    """数据归一化处理"""
    # 计算归一化因子
    regular_{var} = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_{var}'])**2, axis=2)), axis=1
    )
    
    # 应用归一化
    data['data_{var}'] = onp.einsum("jkl,j->jkl", data['data_{var}'], regular_{var})
    data['mc_{var}'] = onp.einsum("jkl,j->jkl", data['mc_{var}'], regular_{var})
    data['truth_{var}'] = onp.einsum("jkl,j->jkl", data['truth_{var}'], regular_{var})
    
    return data

def prepare_data_for_jax(data, device=None):
    """将数据转换为JAX格式并放到指定设备"""
    jax_data = {}
    for key, value in data.items():
        jax_data[key] = device_put(np.array(value), device=device)
    return jax_data