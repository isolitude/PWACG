#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM自动生成的f980共振态计算函数
基于配置文件: agent/resonances_config.toml
生成模型: o3-pro-2025-06-10
生成时间: 2025-08-22 12:36:33

警告：此文件由LLM自动生成，手动修改可能会被覆盖
"""

import jax.numpy as np
from jax import vmap
from functools import partial
from dlib import dplex

# 依赖的基础函数 (需要从其他模块导入)
# from fit_hvp_templates import BW, flatte980, flatte1270



# 参数提取代码
```python
# -*- coding: utf-8 -*-
# Auto-generated PWA code – f0(980) (Flatte) inside φ → K K̄ decay chain
# Author: AI-Code-Gen

import jax.numpy as np
from jax import vmap
from functools import partial
from dlib import dplex  # dplex: JAX compliant complex number helper

# ------------------------------------------------------------
# Physics constants (GeV)
# ------------------------------------------------------------
pi_mass = 0.13957     # charged pion mass
k_mass  = 0.493677    # charged kaon  mass

# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------
def _phase_space_factor(s, m1, m2):
    """
    Two-body phase space factor ρ(s) = sqrt( (1-(m1+m2)^2/s)(1-(m1-m2)^2/s ) )
    Returns 0 below threshold.
    """
    m_sum  = m1 + m2
    m_diff = np.abs(m1 - m2)
    lam    = (s - m_sum**2) * (s - m_diff**2)
    lam    = np.maximum(lam, 0.0)
    return np.where(lam > 0.0, np.sqrt(lam) / s, 0.0)


def _build_complex(c, theta):
    """Return complex coefficient c·e^{iθ}."""
    return dplex(c * np.cos(theta), c * np.sin(theta))

# ------------------------------------------------------------
# Resonance propagators
# ------------------------------------------------------------
def breit_wigner(s, m0, gamma0):
    """
    Constant-width Breit-Wigner: 1 / (m0² − s − i m0 Γ0)
    Implemented with explicit real / imag parts to keep JAX real-valued.
    """
    a   = m0**2 - s
    b   = m0 * gamma0
    den = a**2 + b**2 + 1e-12        # stabiliser
    return dplex(a / den, b / den)


def flatte_980(s, m0, g_kk, rg):
    """
    Flatte line shape for f₀(980):
      1 / (m0² − s − i( g_{ππ} ρ_{ππ}(s) + g_{KK} ρ_{KK}(s) ))
    with g_{ππ} = rg · g_{KK}
    """
    g_pipi = rg * g_kk

    rho_pipi = _phase_space_factor(s, pi_mass,  pi_mass)
    rho_kk   = _phase_space_factor(s, k_mass,  k_mass)

    width    = g_pipi * rho_pipi + g_kk * rho_kk
    a        = m0**2 - s
    b        = width
    den      = a**2 + b**2 + 1e-12
    return dplex(a / den, b / den)


# ------------------------------------------------------------
# === f980 resonance parameter extraction ===
# Based on config: f980 (flatte)
# ------------------------------------------------------------

# Fixed parameters for intermediate φ
phi_mass  = np.array([1.02])   # [GeV]
phi_width = np.array([0.004])  # [GeV]

# Internal indices inside args vector
KK_F980_MASS_IDX  = 0
KK_F980_GKK_IDX   = 1
KK_F980_RG_IDX    = 2
KK_F980_CONST_IDX = 3
KK_F980_THETA_IDX = 4


def _get_f980_params(args, base_idx=0):
    """
    Extract f0(980) parameters from the global optimisation vector `args`.
      base_idx … offset position of kk_f980_mass in args.
    Returns: (m0, g_kk, rg, const2, theta2)
    """
    m0     = args[base_idx + KK_F980_MASS_IDX]   # f0(980) pole mass
    g_kk   = args[base_idx + KK_F980_GKK_IDX]    # coupling to K K̄
    rg     = args[base_idx + KK_F980_RG_IDX]     # g_{ππ}/g_{KK}
    const2 = args[base_idx + KK_F980_CONST_IDX]  # floating real magnitude
    theta2 = args[base_idx + KK_F980_THETA_IDX]  # floating phase (rad)
    return m0, g_kk, rg, const2, theta2


# ------------------------------------------------------------
# Amplitude builders
# ------------------------------------------------------------
@partial(vmap, in_axes=(None, 0, 0, None))
def calculate_BW_flatte980(args, s_phi, s_f, base_idx=0):
    """
    Full chain amplitude:
      A = (C₁ + C₂) · BW_φ(s_φ) · Flatte_f0(s_f)
    Vectorised over s_phi & s_f (1-D arrays of same length).
    """
    # Floating resonance parameters
    m0, g_kk, rg, c2, th2 = _get_f980_params(args, base_idx)

    # Fixed complex coefficient (from configuration)
    coeff1 = _build_complex(0.1, 0.1)       # const1 / theta1 (fixed)
    coeff2 = _build_complex(c2,  th2)       # const2 / theta2 (floating)

    # Propagators
    bw_phi     = breit_wigner(s_phi, phi_mass[0], phi_width[0])
    flatte_f0  = flatte_980(s_f, m0, g_kk, rg)

    return (coeff1 + coeff2) * bw_phi * flatte_f0


def lasso_calculate_BW_flatte980(args, s_phi, s_f,
                                 lasso_lambda=0.0, base_idx=0):
    """
    Same as calculate_BW_flatte980 but with LASSO shrinkage applied
    to the floating magnitude const2.
    """
    m0, g_kk, rg, c2, th2 = _get_f980_params(args, base_idx)

    # Apply LASSO (soft) shrinkage to the floating magnitude
    c2_shrink = c2 * np.maximum(1.0 - lasso_lambda, 0.0)

    coeff1 = _build_complex(0.1,       0.1)
    coeff2 = _build_complex(c2_shrink, th2)

    bw_phi    = breit_wigner(s_phi, phi_mass[0], phi_width[0])
    flatte_f0 = flatte_980(s_f, m0, g_kk, rg)

    return (coeff1 + coeff2) * bw_phi * flatte_f0
```


# 标准计算函数
# 标准函数生成失败: string indices must be integers, not 'str'

# Lasso版本函数 (用于约束计算)
# Lasso函数生成失败: string indices must be integers, not 'str'