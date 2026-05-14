import copy, json, logging, logging.config, os, sys, re, time, glob
from functools import partial
from multiprocessing import Array, Barrier, Lock, Manager, Pipe, Process, Value
from threading import Thread
import jax.numpy as np
import numpy as onp
from dlib import dplex
from jax import device_put
from jax import devices as jdevices
from jax import grad, hessian, jit, jvp, value_and_grad, vmap
from scipy.optimize import minimize
from iminuit import minimize as i_minimize
import pynvml

logger = logging.getLogger("lasso")
with open("config/logconfig_lasso.json", "r") as config:
    LOGGING_CONFIG = json.load(config)
    logging.config.dictConfig(LOGGING_CONFIG)


class args(object):
    def __init__(self):
        self.max_processes = None
        self.max_processes_memory = None
        self.processes_gpus = None
        self.thread_gpus = None
        self.threads_in_one_gpu = None
        self.total_gpu_id = None
        self.data_slices = None
        self.mc_slices = None
        self.mini_run = None
        self.bic_delete_mods = list()


class ProcessReturns:
    def __init__(self):
        self.manager = Manager()
        self._dict = self.manager.dict(data_numbering=None, process_id=None,
            parameter_values=None, parameter_errors=None, min_fcn=None,
            timer=None, gpu_id=None)
    def set(self, key, value): self._dict[key] = value
    def get(self, key): return self._dict[key]
    def info(self):
        logger.info("No. {} data on GPU{} for {} s".format(
            self._dict["data_numbering"], self._dict["gpu_id"], self._dict["timer"]))
        for value, error in zip(self._dict["parameter_values"], self._dict["parameter_errors"]):
            logger.debug("value={}, error={}".format(value, error))


class ProcessInitializers:
    def __init__(self):
        self.data_phi_kk = None; self.mc_phi_kk = None; self.truth_phi_kk = None
        self.data_f_kk = None; self.mc_f_kk = None; self.truth_f_kk = None
        self.data_phif0_kk = None; self.mc_phif0_kk = None; self.truth_phif0_kk = None
        self.data_phif2_kk = None; self.mc_phif2_kk = None; self.truth_phif2_kk = None
        self.wt_data_kk = None
        self.data_numbering = None; self.gpu_id = None
        self.gpu_memory_limit_percentage = None
    def __repr__(self): return "No. " + str(self.data_numbering) + " data batch for process initializing on gpu" + str(self.gpu_id)
    def __enter__(self): return self
    def __exit__(self, type, value, trace): return self


class Process_Initializer_Generator():
    def __init__(self, data_slices=None, mc_slices=None, mini_run=None):
        self.data_slices = data_slices
        self.mc_slices = mc_slices
        self.mini_run = mini_run

    def reader_amp(self, file_name):
        amp_list = list()
        with onp.load(file_name) as amp:
            for amp_name in amp.files:
                amp_list.append((amp[amp_name])[:,0:2])
        return onp.array(amp_list)

    def data_npz(self, num):
        self.data_phi_kk = onp.load("data/real_data/phi_kk.npy")
        self.all_data_phi_kk = onp.array_split(self.data_phi_kk, num, axis=1)
        self.data_f_kk = onp.load("data/real_data/f_kk.npy")
        self.all_data_f_kk = onp.array_split(self.data_f_kk, num, axis=0)
        self.data_phif0_kk = onp.load("data/real_data/phif0_kk.npy")
        self.all_data_phif0_kk = onp.array_split(self.data_phif0_kk, num, axis=1)
        self.data_phif2_kk = onp.load("data/real_data/phif2_kk.npy")
        self.all_data_phif2_kk = onp.array_split(self.data_phif2_kk, num, axis=1)
        wt_data_kk = onp.load("data/weight/weight_kk.npy")
        self.all_wt_data_kk = onp.array_split(wt_data_kk, num, axis=0)

    def mc_npz(self, num):
        self.mc_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_mc_phi_kk = onp.array_split(self.mc_phi_kk, num, axis=1)
        self.mc_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_mc_f_kk = onp.array_split(self.mc_f_kk, num, axis=0)
        self.mc_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_mc_phif0_kk = onp.array_split(self.mc_phif0_kk, num, axis=1)
        self.mc_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_mc_phif2_kk = onp.array_split(self.mc_phif2_kk, num, axis=1)

    def regular(self):
        self.re_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        regular_phi_kk = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phi_kk)**2, axis=2)), axis=1)
        self.all_data_phi_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_data_phi_kk), regular_phi_kk)
        self.all_mc_phi_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_mc_phi_kk), regular_phi_kk)

        self.re_f_kk = onp.load("data/mc_truth/f_kk.npy")
        regular_f_kk = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_f_kk)**2, axis=2)), axis=1)
        self.all_data_f_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_data_f_kk), regular_f_kk)
        self.all_mc_f_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_mc_f_kk), regular_f_kk)

        self.re_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        regular_phif0_kk = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif0_kk)**2, axis=2)), axis=1)
        self.all_data_phif0_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_data_phif0_kk), regular_phif0_kk)
        self.all_mc_phif0_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_mc_phif0_kk), regular_phif0_kk)

        self.re_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        regular_phif2_kk = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif2_kk)**2, axis=2)), axis=1)
        self.all_data_phif2_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_data_phif2_kk), regular_phif2_kk)
        self.all_mc_phif2_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_mc_phif2_kk), regular_phif2_kk)

    def process_initializer_generator(self):
        self.data_npz(self.data_slices)
        self.mc_npz(self.mc_slices)
        self.regular()
        logger.info("============ w i t h  n e x t ===============")
        data_numbering = 0
        event_num = self.mini_run
        while data_numbering < event_num:
            _ = ProcessInitializers()
            _.data_f_kk = self.all_data_f_kk[data_numbering % self.data_slices]
            _.mc_f_kk = self.all_mc_f_kk[data_numbering % self.mc_slices]
            _.data_phi_kk = self.all_data_phi_kk[data_numbering % self.data_slices]
            _.mc_phi_kk = self.all_mc_phi_kk[data_numbering % self.mc_slices]
            _.data_phif0_kk = self.all_data_phif0_kk[data_numbering % self.data_slices]
            _.mc_phif0_kk = self.all_mc_phif0_kk[data_numbering % self.mc_slices]
            _.data_phif2_kk = self.all_data_phif2_kk[data_numbering % self.data_slices]
            _.mc_phif2_kk = self.all_mc_phif2_kk[data_numbering % self.mc_slices]
            _.wt_data_kk = self.all_wt_data_kk[data_numbering % self.data_slices]
            _.data_numbering = data_numbering
            yield _
            data_numbering += 1
        return


class PWAFunc():
    def __init__(self, cdl=None, device_id=None):
        self.device = device_id
        self.data_phi_kk = device_put(np.array(cdl.data_phi_kk), device=self.device)
        self.mc_phi_kk = device_put(np.array(cdl.mc_phi_kk), device=self.device)
        self.data_f_kk = device_put(np.array(cdl.data_f_kk), device=self.device)
        self.mc_f_kk = device_put(np.array(cdl.mc_f_kk), device=self.device)
        self.data_phif0_kk = device_put(np.array(cdl.data_phif0_kk), device=self.device)
        self.mc_phif0_kk = device_put(np.array(cdl.mc_phif0_kk), device=self.device)
        self.data_phif2_kk = device_put(np.array(cdl.data_phif2_kk), device=self.device)
        self.mc_phif2_kk = device_put(np.array(cdl.mc_phif2_kk), device=self.device)
        self.wt_data_kk = device_put(np.array(cdl.wt_data_kk), device=self.device)
        self.truth_phi_kk = device_put(np.array(cdl.truth_phi_kk), device=self.device)
        self.truth_f_kk = device_put(np.array(cdl.truth_f_kk), device=self.device)
        self.truth_phif0_kk = device_put(np.array(cdl.truth_phif0_kk), device=self.device)
        self.truth_phif2_kk = device_put(np.array(cdl.truth_phif2_kk), device=self.device)

    def data_likelihood_kk(self, args):
        # Extracting parameters from args (global parameter vector)
        # phi_mass and phi_width are fixed constants
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        # kk_f980 parameters
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1, args[3]]).reshape(-1,2)
        kk_f980_theta = np.array([0.1, args[4]]).reshape(-1,2)
        # kk_f0 parameters
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7], args[8]]).reshape(-1,2)
        kk_f0_theta = np.array([args[9], args[10]]).reshape(-1,2)
        # kk_f1270 parameters
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1,5)
        kk_f1270_theta = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1,5)
        # kk_f2 parameters
        kk_f2_mass = np.array([args[23], args[24], args[25]])
        kk_f2_width = np.array([args[26], args[27], args[28]])
        kk_f2_const = np.array([args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43]]).reshape(-1,5)
        kk_f2_theta = np.array([args[44], args[45], args[46], args[47], args[48], args[49], args[50], args[51], args[52], args[53], args[54], args[55], args[56], args[57], args[58]]).reshape(-1,5)

        # Calculate amplitudes for data
        data_phif0_kk_BW_flatte980 = self.calculate_BW_flatte980(
            phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta,
            self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        data_phif0_kk_BW_BW = self.calculate_BW_BW(
            phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta,
            self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        data_phif2_kk_BW_flatte1270 = self.calculate_BW_flatte1270(
            phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta,
            self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        data_phif2_kk_BW_BW = self.calculate_BW_BW(
            phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta,
            self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)

        # Lasso amplitudes (use lasso_calculate_* with truth data, preserves l dim)
        lasso_data_phif0_kk_BW_flatte980 = self.lasso_calculate_BW_flatte980(
            phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta,
            self.truth_phi_kk, self.truth_f_kk, self.truth_phif0_kk)
        lasso_data_phif0_kk_BW_BW = self.lasso_calculate_BW_BW(
            phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta,
            self.truth_phi_kk, self.truth_f_kk, self.truth_phif0_kk)
        lasso_data_phif2_kk_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(
            phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta,
            self.truth_phi_kk, self.truth_f_kk, self.truth_phif2_kk)
        lasso_data_phif2_kk_BW_BW = self.lasso_calculate_BW_BW(
            phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta,
            self.truth_phi_kk, self.truth_f_kk, self.truth_phif2_kk)

        step_function = 0.0

        return -np.sum(np.log(np.sum(dplex.dabs(
            data_phif0_kk_BW_flatte980 + data_phif0_kk_BW_BW + data_phif2_kk_BW_flatte1270 + data_phif2_kk_BW_BW
        ), axis=1))) + 100.0 * step_function

    def mc_likelihood_kk(self, args):
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1, args[3]]).reshape(-1,2)
        kk_f980_theta = np.array([0.1, args[4]]).reshape(-1,2)
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7], args[8]]).reshape(-1,2)
        kk_f0_theta = np.array([args[9], args[10]]).reshape(-1,2)
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1,5)
        kk_f1270_theta = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1,5)
        kk_f2_mass = np.array([args[23], args[24], args[25]])
        kk_f2_width = np.array([args[26], args[27], args[28]])
        kk_f2_const = np.array([args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43]]).reshape(-1,5)
        kk_f2_theta = np.array([args[44], args[45], args[46], args[47], args[48], args[49], args[50], args[51], args[52], args[53], args[54], args[55], args[56], args[57], args[58]]).reshape(-1,5)

        mc_phif0_kk_BW_flatte980 = self.calculate_BW_flatte980(
            phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta,
            self.mc_phi_kk, self.mc_f_kk, self.mc_phif0_kk)
        mc_phif0_kk_BW_BW = self.calculate_BW_BW(
            phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta,
            self.mc_phi_kk, self.mc_f_kk, self.mc_phif0_kk)
        mc_phif2_kk_BW_flatte1270 = self.calculate_BW_flatte1270(
            phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta,
            self.mc_phi_kk, self.mc_f_kk, self.mc_phif2_kk)
        mc_phif2_kk_BW_BW = self.calculate_BW_BW(
            phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta,
            self.mc_phi_kk, self.mc_f_kk, self.mc_phif2_kk)

        return np.sum(dplex.dabs(
            mc_phif0_kk_BW_flatte980 + mc_phif0_kk_BW_BW + mc_phif2_kk_BW_flatte1270 + mc_phif2_kk_BW_BW
        ))

    def calculate_BW_flatte980(self, phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, const, theta, sbc_phi, sbc_f, amp):
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        flatte_f = vmap(partial(self.flatte980, Sbc=sbc_f), out_axes=1)(kk_f980_mass, kk_f980_g_kk, kk_f980_rg)
        bw = dplex.deinsum("j,ij->ij", bw_phi, flatte_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def calculate_BW_BW(self, phi_mass, phi_width, mass, width, const, theta, sbc_phi, sbc_f, amp):
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        bw_f = vmap(partial(self.BW, Sbc=sbc_f), out_axes=1)(mass, width)
        bw = dplex.deinsum("j,ij->ij", bw_phi, bw_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def calculate_BW_flatte1270(self, phi_mass, phi_width, mass, width, const, theta, sbc_phi, sbc_f, amp):
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        flatte_f = vmap(partial(self.flatte1270, Sbc=sbc_f), out_axes=1)(mass, width)
        bw = dplex.deinsum("j,ij->ij", bw_phi, flatte_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def lasso_calculate_BW_flatte980(self, phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, const, theta, sbc_phi, sbc_f, amp):
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        flatte_f = vmap(partial(self.flatte980, Sbc=sbc_f), out_axes=1)(kk_f980_mass, kk_f980_g_kk, kk_f980_rg)
        bw = dplex.deinsum("j,ij->ij", bw_phi, flatte_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        return phif

    def lasso_calculate_BW_BW(self, phi_mass, phi_width, mass, width, const, theta, sbc_phi, sbc_f, amp):
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        bw_f = vmap(partial(self.BW, Sbc=sbc_f), out_axes=1)(mass, width)
        bw = dplex.deinsum("j,ij->ij", bw_phi, bw_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        return phif

    def lasso_calculate_BW_flatte1270(self, phi_mass, phi_width, mass, width, const, theta, sbc_phi, sbc_f, amp):
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        flatte_f = vmap(partial(self.flatte1270, Sbc=sbc_f), out_axes=1)(mass, width)
        bw = dplex.deinsum("j,ij->ij", bw_phi, flatte_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        return phif

    def BW(self, m_, w_, Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc, -m_*w_*np.ones(l))
        return dplex.ddivide(1.0, temp)

    def flatte980(self, m_, g_pipi, rg, Sbc):
        g_kk = rg * g_pipi
        m_k = 0.493677; m_pi = 0.13957061
        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        tmp_A = dplex.dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
        return dplex.ddivide(1.0, tmp_A)

    def flatte1270(self, m_, w_, Sbc):
        rm = m_ * m_; gr = m_ * w_
        q2r = 0.25 * rm - 0.0194792
        b2r = q2r * (q2r + 0.1825) + 0.033306
        g11270 = gr * b2r / np.power(q2r, 2.5)
        q2 = 0.25 * Sbc - 0.0194792
        b2 = q2 * (q2 + 0.1825) + 0.033306
        g1 = g11270 * np.power(q2, 2.5) / b2
        tmp = dplex.dconstruct(Sbc - rm, g1)
        return dplex.ddivide(gr, tmp)

    def hvp_data_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.data_likelihood_kk), [args_float], [any_vector])

    def hvp_mc_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.mc_likelihood_kk), [args_float], [any_vector])

    def jit_request(self):
        self.jit_data_likelihood_kk = jit(self.data_likelihood_kk, device=self.device)
        self.jit_mc_likelihood_kk = jit(self.mc_likelihood_kk, device=self.device)
        self.jit_grad_data_likelihood_kk = jit(grad(self.data_likelihood_kk), device=self.device)
        self.jit_grad_mc_likelihood_kk = jit(value_and_grad(self.mc_likelihood_kk), device=self.device)
        self.jit_hvp_data_fwdrev_kk = jit(self.hvp_data_fwdrev_kk, device=self.device)
        self.jit_hvp_mc_fwdrev_kk = jit(self.hvp_mc_fwdrev_kk, device=self.device)


class Control(object):
    def __init__(self, args):
        self.bic_delete_mods = args.bic_delete_mods
        self.max_processes = args.max_processes
        self.max_processes_memory = args.max_processes_memory
        self.processes_gpus = args.processes_gpus
        self.thread_gpus = args.thread_gpus
        self.threads_in_one_gpu = args.threads_in_one_gpu
        self.total_gpu_id = args.total_gpu_id
        self.data_slices = args.data_slices
        self.mc_slices = args.mc_slices
        self.mini_run = args.mini_run

        # Global parameter list with initial values (dummy, will be overwritten by fit)
        # We have 59 float parameters (indices 0-58)
        self.args_list = onp.array([0.98, 0.5, 1.0, 0.5, 1.0, 1.704, 0.123, 1.0, 1.0, 0.0, 0.0,
                                     1.2755, 0.1867, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     1.517, 2.157, 2.345, 0.086, 0.152, 0.322, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # All parameters are float in lasso
        self.float_list = onp.arange(59)

        max_processes = self.max_processes
        self.process_pool = [Process()] * max_processes
        self.kk_data_lh = onp.zeros([self.thread_gpus, self.threads_in_one_gpu])
        self.kk_mc_lh = onp.zeros([self.thread_gpus, self.threads_in_one_gpu])
        self.kk_data_grad = onp.zeros([self.thread_gpus, self.threads_in_one_gpu, self.float_list.shape[0]])
        self.kk_mc_grad = onp.zeros([self.thread_gpus, self.threads_in_one_gpu, self.float_list.shape[0]])
        self.kk_mc_grad_v = onp.zeros([self.thread_gpus, self.threads_in_one_gpu])
        self.kk_data_hvp = onp.zeros([self.thread_gpus, self.threads_in_one_gpu, self.float_list.shape[0]])
        self.kk_mc_hvp = onp.zeros([self.thread_gpus, self.threads_in_one_gpu, self.float_list.shape[0]])
        self.kk_mc_hvp_g = onp.zeros([self.thread_gpus, self.threads_in_one_gpu, self.float_list.shape[0]])
        self.kk_mc_hvp_v = onp.zeros([self.thread_gpus, self.threads_in_one_gpu])

    def likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
            self.kk_data_lh[device_rank, _] = self.pwaf_list[device_rank][_].jit_data_likelihood_kk(args_float)
            self.kk_mc_lh[device_rank, _] = self.pwaf_list[device_rank][_].jit_mc_likelihood_kk(args_float)

    def thread_likelihood(self, args_float):
        threading_list = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target=self.likelihood_in_sigle_device, args=(args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus): threading_list[_].start()
        for _ in range(self.thread_gpus): threading_list[_].join()
        result_kk = onp.sum(self.kk_data_lh) + 100.0 * onp.log(onp.sum(self.kk_mc_lh) / 1000.0)
        return result_kk

    def jit_grad_likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
            self.kk_data_grad[device_rank, _, :] = self.pwaf_list[device_rank][_].jit_grad_data_likelihood_kk(args_float)
            self.kk_mc_grad_v[device_rank, _], self.kk_mc_grad[device_rank, _, :] = self.pwaf_list[device_rank][_].jit_grad_mc_likelihood_kk(args_float)

    def thread_grad_likelihood(self, args_float):
        threading_list = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target=self.jit_grad_likelihood_in_sigle_device, args=(args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus): threading_list[_].start()
        for _ in range(self.thread_gpus): threading_list[_].join()
        result_kk = onp.sum(self.kk_data_grad, axis=(0,1)) + 100.0 * onp.sum(self.kk_mc_grad, axis=(0,1)) / onp.sum(self.kk_mc_grad_v)
        return result_kk

    def thread_hvp(self, args_float, any_vector):
        # For simplicity, implement a numerically approximated HVP using finite difference of grad
        eps = 1e-5
        grad_plus = self.thread_grad_likelihood(args_float + eps * any_vector)
        grad_minus = self.thread_grad_likelihood(args_float - eps * any_vector)
        return (grad_plus - grad_minus) / (2 * eps)

    def compile_func(self):
        vector = onp.ones(self.float_list.shape)
        # Enable XLA compilation by running once
        self.thread_likelihood(self.args_list[self.float_list])
        self.thread_grad_likelihood(self.args_list[self.float_list])
        self.thread_hvp(self.args_list[self.float_list], vector)

    def run(self, process_initializer, process_returns):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x:str(x), process_initializer[0][0].total_gpu_id))
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(process_initializer[0][0].gpu_memory_limit_percentage)
        from jax.config import config
        config.update("jax_enable_x64", True)
        device_list = jdevices()
        self.pwaf_list = [[PWAFunc(process_initializer[i][j], device_list[process_initializer[i][j].gpu_id])
                          for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
        for i in range(self.thread_gpus):
            for j in range(self.threads_in_one_gpu):
                self.pwaf_list[i][j].jit_request()
        start_time = time.time()
        num_seed = onp.random.randint(low=0, high=1000000, size=1, dtype='l')
        logger.info("random seed: {}".format(num_seed))
        onp.random.seed(num_seed)
        self.args_float = self.args_list[self.float_list]
        theta_index = onp.array([4, 10, 22, 58])  # indices of theta parameters in the global list (kk_f980_theta, kk_f0_theta, kk_f1270_theta, kk_f2_theta)
        const_index = onp.array([3, 7, 8, 13, 14, 15, 16, 17, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43])  # all const indices
        self.compile_func()
        min_fcn = 0.0
        result_info = dict()
        for n in range(1):  # number of cycles, assumed 1 from context
            args_float = self.args_float
            t1 = time.time()
            res = minimize(fun=self.thread_likelihood, x0=args_float,
                          jac=self.thread_grad_likelihood, hessp=self.thread_hvp,
                          method="Newton-CG", callback=self.my_callback,
                          options={"disp": False, "xtol": 1e-8})
            t2 = time.time()
            logger.info("Cycle {} finished in {:.2f} s".format(n, t2 - t1))
            # Extract results
            fvalue = res.x
            ferror = onp.sqrt(onp.diag(onp.linalg.inv(res.hess_inv)))
            min_fcn = res.fun
            result_info[n] = {"value": fvalue.tolist(), "error": ferror.tolist(), "fcn": min_fcn}
            self.save_in_json(fvalue, ferror, "output/fit/fit_result_kk/pwa_info_kk_lasso", str(n), result_info)
        process_returns.set("parameter_values", fvalue)
        process_returns.set("parameter_errors", ferror)
        process_returns.set("min_fcn", min_fcn)
        stop_time = time.time()
        process_returns.info()
        process_returns.set("timer", stop_time - start_time)
        process_returns.set("process_id", os.getpid())
        process_returns.set("gpu_id", process_initializer[0][0].total_gpu_id)

    def run_multiprocess(self):
        process_initializer = Process_Initializer_Generator(
            data_slices=self.data_slices,
            mc_slices=self.mc_slices,
            mini_run=self.mini_run)
        max_processes = self.max_processes
        process_gen = process_initializer.process_initializer_generator()
        process_returns_list = [ProcessReturns() for _ in range(max_processes)]
        process_pool = []
        for i in range(max_processes):
            try:
                p_initializer = [[next(process_gen) for _ in range(self.thread_gpus * self.threads_in_one_gpu)]]
                p_initializer[0][0].total_gpu_id = self.total_gpu_id
                p_initializer[0][0].gpu_memory_limit_percentage = self.max_processes_memory
                for k in range(self.thread_gpus * self.threads_in_one_gpu):
                    p_initializer[0][k].gpu_id = k
                p = Process(target=self.run, args=(p_initializer, process_returns_list[i]))
                process_pool.append(p)
                p.start()
            except StopIteration:
                break
        for p in process_pool:
            p.join()
        for pr in process_returns_list:
            if pr.get("parameter_values") is not None:
                logger.info("Final parameter values: {}".format(pr.get("parameter_values")))
                logger.info("Final parameter errors: {}".format(pr.get("parameter_errors")))

    def save_in_json(self, np_values, np_errors, save_addr="", n="", result_info={}):
        os.makedirs(os.path.dirname(save_addr), exist_ok=True)
        data = {"parameters": np_values.tolist(), "errors": np_errors.tolist(), "iteration": n, "result_info": result_info}
        with open(save_addr + "_" + str(n) + ".json", "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved fit results to {}".format(save_addr))

    def my_callback(self, xk):
        pass