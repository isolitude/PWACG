#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM自动生成的f980共振态计算函数
基于配置文件: agent/resonances_config.toml
生成模型: gpt-5-2025-08-07
生成时间: 2025-08-28 11:01:48

警告：此文件由LLM自动生成，手动修改可能会被覆盖
"""

import jax.numpy as np
from jax import vmap
from functools import partial
from dlib import dplex

# 依赖的基础函数 (需要从其他模块导入)
# from fit_hvp_templates import BW, flatte980, flatte1270



# 标准计算函数
from functools import partial
import jax.numpy as np
from jax import vmap
from dlib import dplex

def calculate_BW_flatte(
    phi_mass, phi_width,
    f_mass, g_kk, rg,
    const, theta,
    phi_kk, f_kk,
    phif0_kk, phif2_kk
):
    """
    Compute the resonance amplitude for the mode 'phif0_980_kk' using a BW(phi) * Flatté(f0(980)) propagator chain.
    - phi: constant-width Breit-Wigner
    - f0(980): two-channel Flatté with pi-pi and K-K thresholds
    The result is coefficient-weighted and applied to the provided phif0_kk base amplitude.
    Returns complex amplitude with shape [j, k], where j = len(const) and k = len(events).
    """
    tiny = 1e-16

    # Two-body equal-mass phase space factor rho(s) with correct analytic continuation
    def _rho_equal_mass_s(s, m):
        # rho(s) = 2 * sqrt(1 - 4 m^2 / s) above threshold; = 2i * sqrt(4 m^2 / s - 1) below.
        s = np.maximum(s, tiny)
        ratio = (4.0 * m * m) / s
        cond = ratio < 1.0
        re = np.where(cond, 2.0 * np.sqrt(np.maximum(1.0 - ratio, 0.0)), 0.0)
        im = np.where(cond, 0.0, 2.0 * np.sqrt(np.maximum(ratio - 1.0, 0.0)))
        return dplex(re, im)

    # Constant-width Breit-Wigner for phi(1020): 1 / (m0^2 - s - i m0 Gamma)
    def _bw_phi_elem(m0, g0, m):
        s = np.square(np.maximum(m, 0.0))
        denom = dplex(m0 * m0 - s, -m0 * g0)
        return dplex(1.0, 0.0) / denom

    bw_phi = partial(vmap, in_axes=(None, None, 0))(_bw_phi_elem)

    # Flatté for f0(980): 1 / (m0^2 - s - i [g_pi rho_pi(s) + g_K rho_K(s)])
    def _flatte_elem(m0, gk, rg_in, m):
        s = np.square(np.maximum(m, 0.0))
        m_pi = 0.13957
        m_k = 0.493677
        rho_pi = _rho_equal_mass_s(s, m_pi)
        rho_k = _rho_equal_mass_s(s, m_k)
        g_pi = rg_in * gk
        width_term = g_pi * rho_pi + gk * rho_k  # complex
        denom = dplex(m0 * m0 - s, 0.0) - dplex(0.0, 1.0) * width_term
        return dplex(1.0, 0.0) / denom

    flatte = partial(vmap, in_axes=(None, None, None, 0))(_flatte_elem)

    # Build complex coefficients: c_j = const_j * exp(i theta_j)
    const = np.asarray(const)
    theta = np.asarray(theta)
    c_j = const * dplex(np.cos(theta), np.sin(theta))  # shape [j]

    # Kinematics arrays (event-wise masses)
    phi_kk = np.asarray(phi_kk)
    f_kk = np.asarray(f_kk)
    k = phi_kk.shape[0]

    # Propagators over events
    bw_k = bw_phi(phi_mass, phi_width, phi_kk)            # shape [k], complex
    flatte_k = flatte(f_mass, g_kk, rg, f_kk)             # shape [k], complex
    chain_k = bw_k * flatte_k                             # combined phi⊗f0 chain

    # Base amplitude for this wave (phif0_kk). Ensure [j, k] then make complex.
    j = c_j.shape[0]
    base_real = np.broadcast_to(np.asarray(phif0_kk), (j, k))
    base_c = dplex(base_real, np.zeros_like(base_real))

    # Apply coefficients and propagators: A[j,k] = c_j * chain_k * base_jk
    amp = (c_j[:, None] * chain_k[None, :]) * base_c

    return amp

# Lasso版本函数 (用于约束计算)
from functools import partial
from jax import vmap
import jax.numpy as np
from dlib import dplex

def lasso_calculate_BW_flatte(m_phi, m_kk, params=None):
    """
    Compute the l-preserving resonance amplitude for phi -> (f0(980) -> K K) using:
    - phi propagator: Breit-Wigner with constant width
    - f0(980) propagator: Flatté line shape (pi-pi and K-K coupled channels)
    The final Einstein multiplication preserves l dimension: ljk, lj -> ljk.
    This is used for branching ratio constraints.

    Inputs
    - m_phi: array-like, phi candidate invariant mass, shape [j] or [j, k]
    - m_kk: array-like, K K invariant mass for f0(980), same shape as m_phi
    - params: dict with optional overrides
        mass: f0 mass (default 0.979481)
        g_kk: f0 coupling to K K (default 0.106786)
        rg: ratio g_pi / g_K (default 8.570188) so g_pi = rg * g_kk
        const1, theta1: fixed complex coefficient (defaults 0.1, 0.1)
        const2, theta2: floating complex coefficient (defaults 0.106518, 0.038070)
        phi_mass: phi pole mass (default 1.02)
        phi_width: phi width (default 0.004)

    Returns
    - complex array with shape [l, j, k], l=2 for two complex coefficients.
    """
    # Defaults from configuration
    if params is None:
        params = {}
    m0_f = params.get('mass', 0.979481)
    g_kk = params.get('g_kk', 0.106786)
    rg = params.get('rg', 8.570188)  # g_pi = rg * g_kk

    const1 = params.get('const1', 0.100000)
    theta1 = params.get('theta1', 0.100000)
    const2 = params.get('const2', 0.106518)
    theta2 = params.get('theta2', 0.038070)

    phi_m0 = params.get('phi_mass', 1.02)
    phi_g0 = params.get('phi_width', 0.004)

    # Physical masses (GeV)
    m_pi = 0.13957039
    m_K = 0.493677

    # Ensure inputs are arrays with shape [j, k]
    m_phi = np.asarray(m_phi)
    m_kk = np.asarray(m_kk)
    if m_phi.ndim == 1:
        m_phi = m_phi[:, None]
    if m_kk.ndim == 1:
        m_kk = m_kk[:, None]
    # Basic shape check
    if m_phi.shape != m_kk.shape:
        raise ValueError("m_phi and m_kk must have the same shape [j, k].")

    j, k = m_phi.shape

    tiny = 1e-12

    # Helper: two-body phase space rho(s) = 2k/sqrt(s) with analytic continuation below threshold
    def _rho_two_body(m, s):
        # s and returns complex rho
        s_c = dplex(s, 0.0)
        sqrt_s = np.sqrt(s_c) + dplex(tiny, 0.0)
        k_sq = s - 4.0 * m * m
        k_c = 0.5 * np.sqrt(dplex(k_sq, 0.0))
        return 2.0 * k_c / sqrt_s

    # Breit-Wigner with constant width for vector phi
    def _bw_const_width(m, m0, gamma0):
        s = m * m
        # Denominator: (m0^2 - s) - i m0 Γ0
        denom = dplex(m0 * m0 - s, -m0 * gamma0 - 0.0)  # small -0.0 to keep dtype
        # Numerical stabilizer to avoid exact zeros on rare coincidences
        denom = denom + dplex(0.0, tiny)
        return 1.0 / denom

    # Flatté for f0(980) with pi-pi and K-K channels
    def _flatte_980(m, m0, gkk, rg_local, mpi, mK):
        s = m * m
        rho_pi = _rho_two_body(mpi, s)
        rho_K = _rho_two_body(mK, s)
        gpi = rg_local * gkk
        # Denominator: (m0^2 - s) - i (g_pi rho_pi + g_K rho_K)
        width_term = (gpi * rho_pi + gkk * rho_K)
        denom = dplex(m0 * m0 - s, 0.0) - dplex(0.0, 1.0) * width_term
        denom = denom + dplex(0.0, tiny)
        return 1.0 / denom

    # Event-wise amplitude (scalar inputs)
    bw_phi_ev = partial(_bw_const_width, m0=phi_m0, gamma0=phi_g0)
    flatte_ev = partial(_flatte_980, m0=m0_f, gkk=g_kk, rg_local=rg, mpi=m_pi, mK=m_K)

    def _event_amp(mphi, mkk):
        return bw_phi_ev(mphi) * flatte_ev(mkk)

    # Vectorize over events: flatten to 1D then reshape
    mphi_vec = m_phi.reshape(-1)
    mkk_vec = m_kk.reshape(-1)
    amp_vec = vmap(_event_amp, in_axes=(0, 0))(mphi_vec, mkk_vec)
    amp_jk = amp_vec.reshape(j, k)

    # Build l-dimension amplitudes by broadcasting; l=2 here
    amp_ljk = np.broadcast_to(amp_jk[None, ...], (2, j, k))

    # Production complex coefficients: c = const * exp(i theta)
    def _make_coeff(mod, phase):
        return dplex(mod * np.cos(phase), mod * np.sin(phase))

    c1 = _make_coeff(const1, theta1)
    c2 = _make_coeff(const2, theta2)
    coeff_l = np.stack([c1, c2], axis=0)  # shape [l]
    # Expand to [l, j] to preserve l and j in the final einsum
    coeff_lj = coeff_l[:, None] * np.ones((1, j))

    # Final multiplication preserving l: ljk, lj -> ljk
    result_ljk = np.einsum('ljk,lj->ljk', amp_ljk, coeff_lj)

    return result_ljk