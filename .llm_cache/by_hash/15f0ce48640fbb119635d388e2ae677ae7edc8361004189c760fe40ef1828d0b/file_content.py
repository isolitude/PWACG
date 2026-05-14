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
import ROOT


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
        self.data_phi_kk = None
        self.data_f_kk = None
        self.data_phif0_kk = None
        self.data_phif2_kk = None
        self.mc_phi_kk = None
        self.mc_f_kk = None
        self.mc_phif0_kk = None
        self.mc_phif2_kk = None
        self.wt_data_kk = None
        self.data_numbering = None
        self.gpu_id = None
        self.gpu_memory_limit_percentage = None
    def __repr__(self):
        return "No. " + str(self.data_numbering) + " data batch for process initializing on gpu" + str(self.gpu_id)
    def __enter__(self):
        return self
    def __exit__(self, type, value, trace):
        return self


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
        self.all_data_f_kk = onp.array_split(self.data_f_kk, num, axis=1)
        self.data_phif0_kk = onp.load("data/real_data/phif0_kk.npy")
        self.all_data_phif0_kk = onp.array_split(self.data_phif0_kk, num, axis=1)
        self.data_phif2_kk = onp.load("data/real_data/phif2_kk.npy")
        self.all_data_phif2_kk = onp.array_split(self.data_phif2_kk, num, axis=1)
        data_f_kk = onp.load("data/real_data/f_kk.npy")
        self.all_data_f_kk = onp.array_split(data_f_kk, num, axis=0)
        data_phi_kk = onp.load("data/real_data/phi_kk.npy")
        self.all_data_phi_kk = onp.array_split(data_phi_kk, num, axis=0)
        wt_data_kk = onp.load("data/weight/weight_kk.npy")
        self.all_wt_data_kk = onp.array_split(wt_data_kk, num, axis=0)

    def mc_npz(self, num):
        self.mc_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_mc_phi_kk = onp.array_split(self.mc_phi_kk, num, axis=1)
        self.mc_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_mc_f_kk = onp.array_split(self.mc_f_kk, num, axis=1)
        self.mc_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_mc_phif0_kk = onp.array_split(self.mc_phif0_kk, num, axis=1)
        self.mc_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_mc_phif2_kk = onp.array_split(self.mc_phif2_kk, num, axis=1)
        mc_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_mc_f_kk = onp.array_split(mc_f_kk, num, axis=0)
        mc_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_mc_phi_kk = onp.array_split(mc_phi_kk, num, axis=0)

    def regular(self):
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

    def data_likelihood_kk(self, args):
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1, args[3]]).reshape(-1, 2)
        kk_f980_theta = np.array([0.1, args[4]]).reshape(-1, 2)
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7], args[8]]).reshape(-1, 2)
        kk_f0_theta = np.array([args[9], args[10]]).reshape(-1, 2)
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5)
        kk_f1270_theta = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5)
        kk_f2_mass = np.array([args[23], args[24], args[25]])
        kk_f2_width = np.array([args[26], args[27], args[28]])
        kk_f2_const = np.array([args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43]]).reshape(-1, 5)
        kk_f2_theta = np.array([args[44], args[45], args[46], args[47], args[48], args[49], args[50], args[51], args[52], args[53], args[54], args[55], args[56], args[57], args[58]]).reshape(-1, 5)

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

        step_function = 0.0
        return -np.sum(np.log(np.sum(dplex.dabs(data_phif0_kk_BW_flatte980 + data_phif0_kk_BW_BW + data_phif2_kk_BW_flatte1270 + data_phif2_kk_BW_BW), axis=1))) + 100.0 * step_function

    def mc_likelihood_kk(self, args):
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1, args[3]]).reshape(-1, 2)
        kk_f980_theta = np.array([0.1, args[4]]).reshape(-1, 2)
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7], args[8]]).reshape(-1, 2)
        kk_f0_theta = np.array([args[9], args[10]]).reshape(-1, 2)
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5)
        kk_f1270_theta = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5)
        kk_f2_mass = np.array([args[23], args[24], args[25]])
        kk_f2_width = np.array([args[26], args[27], args[28]])
        kk_f2_const = np.array([args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43]]).reshape(-1, 5)
        kk_f2_theta = np.array([args[44], args[45], args[46], args[47], args[48], args[49], args[50], args[51], args[52], args[53], args[54], args[55], args[56], args[57], args[58]]).reshape(-1, 5)

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

        return np.sum(dplex.dabs(mc_phif0_kk_BW_flatte980 + mc_phif0_kk_BW_BW + mc_phif2_kk_BW_flatte1270 + mc_phif2_kk_BW_BW))

    def calculate_BW_flatte980(self, phi_mass, phi_width, mass, g_kk, rg, const, theta, sbc_phi, sbc_f, amp):
        bw = self.BW_flatte980(phi_mass, phi_width, mass, g_kk, rg, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def calculate_BW_flatte1270(self, phi_mass, phi_width, mass, width, const, theta, sbc_phi, sbc_f, amp):
        bw = self.BW_flatte1270(phi_mass, phi_width, mass, width, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def calculate_BW_BW(self, phi_mass, phi_width, mass, width, const, theta, sbc_phi, sbc_f, amp):
        bw = self.BW_BW(phi_mass, phi_width, mass, width, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(const, theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def BW_flatte980(self, phi_mass, phi_width, mass, g_kk, rg, sbc_phi, sbc_f):
        a = self.BW(phi_mass, phi_width, sbc_phi)
        b = vmap(partial(self.flatte980, Sbc=sbc_f), out_axes=1)(mass, g_kk, rg)
        return dplex.deinsum("j, ij->ij", a, b)

    def BW_flatte1270(self, phi_mass, phi_width, mass, width, sbc_phi, sbc_f):
        a = self.BW(phi_mass, phi_width, sbc_phi)
        b = vmap(partial(self.flatte1270, Sbc=sbc_f), out_axes=1)(mass, width)
        return dplex.deinsum("j, ij->ij", a, b)

    def BW_BW(self, phi_mass, phi_width, mass, width, sbc_phi, sbc_f):
        a = self.BW(phi_mass, phi_width, sbc_phi)
        b = vmap(partial(self.BW_fside, Sbc=sbc_f), out_axes=1)(mass, width)
        return dplex.deinsum("j, ij->ij", a, b)

    def BW(self, m_, w_, Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_ * m_ - Sbc, -m_ * w_ * np.ones(l))
        return dplex.ddivide(1.0, temp)

    def BW_fside(self, m_, w_, Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_ * m_ - Sbc, -m_ * w_ * np.ones(l))
        return dplex.ddivide(1.0, temp)

    def BW_relativity(self, m_, w_, Sbc):
        gamma = np.sqrt(m_ * m_ * (m_ * m_ + w_ * w_))
        k = np.sqrt(2 * np.sqrt(2) * m_ * np.abs(w_) * gamma / np.pi / np.sqrt(m_ * m_ + gamma))
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_ * m_ - Sbc, -m_ * w_ * np.ones(l))
        return dplex.ddivide(k, temp)

    def flatte980(self, m_, g_pipi, rg, Sbc):
        g_kk = rg * g_pipi
        m_k = 0.493677
        m_pi = 0.13957061
        rho_kk = np.sqrt(np.abs(1 - 4 * m_k * m_k / Sbc))
        rho_pipi = np.sqrt(np.abs(1 - 4 * m_pi * m_pi / Sbc))
        tmp_A = dplex.dconstruct(m_ ** 2 - Sbc, -1 * (g_pipi * rho_pipi + g_kk * rho_kk))
        return dplex.ddivide(1.0, tmp_A)

    def flatte1270(self, m_, w_, Sbc):
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

    def flatte500(self, m_, b1, b2, b3, b4, b5, Sbc):
        m2 = m_ * m_
        rp = 0.139556995
        mpi2d2 = 0.009739946882
        cro1 = np.sqrt(np.abs((Sbc - (2 * rp) ** 2) * Sbc)) / Sbc
        cro2 = np.sqrt(np.abs((m2 - (2 * rp) ** 2) * m2)) / m2
        pip1 = np.sqrt(np.abs(1.0 - 0.3116765584 / Sbc)) / (1.0 + np.exp(9.8 - 3.5 * Sbc))
        pip2 = np.sqrt(np.abs(1.0 - 0.3116765584 / m2)) / (1.0 + np.exp(9.8 - 3.5 * m2))
        cgam1 = m_ * (b1 + b2 * Sbc) * (Sbc - mpi2d2) / (m2 - mpi2d2) * np.exp(-(Sbc - m2) / b3) * cro1 / cro2
        cgam2 = m_ * b4 * pip1 / pip2
        tmp = dplex.dconstruct(m2 - Sbc, -b5 * (cgam1 + cgam2))
        return dplex.ddivide(1.0, tmp)

    def hvp_data_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.data_likelihood_kk), [args_float], [any_vector])

    def hvp_mc_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.mc_likelihood_kk), [args_float], [any_vector])

    def phase(self, theta):
        return vmap(self._phase)(theta)

    def _phase(self, theta):
        return dplex.dconstruct(np.cos(theta), np.sin(theta))

    def mod_weight(self, mod_name, args_list, float_list, const_index, num_iamps, all_const_index):
        for n, i in enumerate(range(0, len(const_index), num_iamps)):
            _args_list = copy.deepcopy(args_list)
            _const_index = const_index[i:i + num_iamps]
            leftover = [x for x in all_const_index if x not in _const_index]
            for i in leftover:
                _args_list[i] = -100.0
            args_float = onp.array(_args_list[float_list])
            wt = self.jit_weight_kk(args_float)
            frac = onp.sum(wt) / self.sum_wt
            self.frac_list.append(frac)

    def run_weight(self, args_list, float_list, bic_index):
        self.jit_request()
        all_const_index = (5, 6, 11, 12, 17, 18, 19, 20, 21, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)
        args_float = onp.array(args_list[float_list])
        args_float[bic_index] = -100.0
        total_wt = self.jit_weight_kk(args_float)
        self.sum_wt = onp.sum(total_wt)
        self.frac_list = list()
        self.mod_weight("phif0_kk_BW_flatte980", args_list, float_list, onp.array([5, 6]), 2, all_const_index)
        self.mod_weight("phif0_kk_BW_BW", args_list, float_list, onp.array([11, 12]), 2, all_const_index)
        self.mod_weight("phif2_kk_BW_flatte1270", args_list, float_list, onp.array([17, 18, 19, 20, 21]), 5, all_const_index)
        self.mod_weight("phif2_kk_BW_BW", args_list, float_list, onp.array([33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]), 5, all_const_index)
        result = self.frac_list
        return result

    def weight_kk(self, args):
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1, args[3]]).reshape(-1, 2)
        kk_f980_theta = np.array([0.1, args[4]]).reshape(-1, 2)
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7], args[8]]).reshape(-1, 2)
        kk_f0_theta = np.array([args[9], args[10]]).reshape(-1, 2)
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5)
        kk_f1270_theta = np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5)
        kk_f2_mass = np.array([args[23], args[24], args[25]])
        kk_f2_width = np.array([args[26], args[27], args[28]])
        kk_f2_const = np.array([args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43]]).reshape(-1, 5)
        kk_f2_theta = np.array([args[44], args[45], args[46], args[47], args[48], args[49], args[50], args[51], args[52], args[53], args[54], args[55], args[56], args[57], args[58]]).reshape(-1, 5)

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

        return np.sum(dplex.dabs(data_phif0_kk_BW_flatte980 + data_phif0_kk_BW_BW + data_phif2_kk_BW_flatte1270 + data_phif2_kk_BW_BW), axis=1)

    def jit_request(self):
        self.jit_data_likelihood_kk = jit(self.data_likelihood_kk, device=self.device)
        self.jit_mc_likelihood_kk = jit(self.mc_likelihood_kk, device=self.device)
        self.jit_weight_kk = jit(self.weight_kk, device=self.device)


class Control(object):
    def __init__(self, args):
        self.bic_delete_mods = args.bic_delete_mods
        self.max_processes = args.max_processes
        self.max_processes_memory = args.max_processes_memory
        self.processes_gpus = args.processes_gpus
        self.thread_gpus = args.thread_gpus
        self.threads_in_one_gpu = args.threads_in_one_gpu
        self.total_gpu_id = args.total_gpu_id
        self.data_config = dict()
        self.data_config["data_slices"] = args.data_slices
        self.data_config["mc_slices"] = args.mc_slices
        self.data_config["mini_run"] = args.mini_run

        param_names = ["kk_f980_mass", "kk_f980_g_kk", "kk_f980_rg", "kk_f980_const", "kk_f980_theta",
                       "kk_f0_mass", "kk_f0_width", "kk_f0_const", "kk_f0_theta",
                       "kk_f1270_mass", "kk_f1270_width", "kk_f1270_const", "kk_f1270_theta",
                       "kk_f2_mass", "kk_f2_width", "kk_f2_const", "kk_f2_theta"]
        n_params = 59
        self.args_list = onp.zeros(n_params)
        self.float_list = onp.arange(n_params)

        max_processes = self.max_processes
        self.process_pool = [Process()] * max_processes
        self.kk_data_lh = onp.zeros([self.thread_gpus, self.threads_in_one_gpu])
        self.kk_mc_lh = onp.zeros([self.thread_gpus, self.threads_in_one_gpu])

    def likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
            self.kk_data_lh[device_rank, _] = self.pwaf_list[device_rank][_].jit_data_likelihood_kk(args_float)
            self.kk_mc_lh[device_rank, _] = self.pwaf_list[device_rank][_].jit_mc_likelihood_kk(args_float)

    def thread_likelihood(self, args_float):
        threading_list = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target=self.likelihood_in_sigle_device, args=(args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
            threading_list[_].start()
        for _ in range(self.thread_gpus):
            threading_list[_].join()
        result_kk = onp.sum(self.kk_data_lh) + 100.0 * onp.log(onp.sum(self.kk_mc_lh) / 1000.0)
        result = result_kk
        return result

    def compile_func(self):
        threading_list = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target=self.likelihood_in_sigle_device,
                                       args=(onp.zeros(len(self.float_list)), _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
            threading_list[_].start()
        for _ in range(self.thread_gpus):
            threading_list[_].join()

    def _load_pwa_info(self):
        info_files = sorted(glob.glob("output/fit/fit_result_kk/pwa_info_kk_*.json"))
        if not info_files:
            logger.error("No pwa_info files found in output/fit/fit_result_kk/")
            return None
        with open(info_files[-1], "r") as f:
            pwa_info = json.load(f)
        return pwa_info

    def run(self, process_initializer, process_returns):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x: str(x), process_initializer[0][0].total_gpu_id))
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(process_initializer[0][0].gpu_memory_limit_percentage)
        config.update("jax_enable_x64", True)
        device_list = jdevices()
        self.pwaf_list = [[PWAFunc(process_initializer[i][j], device_list[process_initializer[i][j].gpu_id])
                           for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
        for i in range(self.thread_gpus):
            for j in range(self.threads_in_one_gpu):
                self.pwaf_list[i][j].jit_request()

        pwa_info = self._load_pwa_info()
        if pwa_info is None:
            logger.error("Could not load pwa_info, using default args_list")
            args_float = self.args_list[self.float_list]
        else:
            param_values = pwa_info.get("parameter_values", {})
            for i, name in enumerate(
                ["kk_f980_mass", "kk_f980_g_kk", "kk_f980_rg", "kk_f980_const", "kk_f980_theta",
                 "kk_f0_mass", "kk_f0_width", "kk_f0_const", "kk_f0_theta",
                 "kk_f1270_mass", "kk_f1270_width", "kk_f1270_const", "kk_f1270_theta",
                 "kk_f2_mass", "kk_f2_width", "kk_f2_const", "kk_f2_theta"]):
                if name in param_values:
                    val = param_values[name]
                    if isinstance(val, list):
                        for j, v in enumerate(val):
                            idx = i + j
                            if idx < len(self.args_list):
                                self.args_list[idx] = v
                    else:
                        if i < len(self.args_list):
                            self.args_list[i] = val
            args_float = self.args_list[self.float_list]

        self.compile_func()
        likelihood = self.thread_likelihood(args_float)
        logger.info("Likelihood for plotting: {}".format(likelihood))

        pwaf = self.pwaf_list[0][0]
        float_list = self.float_list
        args_list = self.args_list
        all_const_index = (5, 6, 11, 12, 17, 18, 19, 20, 21, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)
        args_float_wt = onp.array(args_list[float_list])
        total_wt_data = onp.asarray(pwaf.jit_weight_kk(args_float_wt))
        sum_wt = onp.sum(total_wt_data)
        frac_list = []
        for mod_idx, (mod_name, const_index, num_iamps) in enumerate([
            ("phif0_kk_BW_flatte980", onp.array([5, 6]), 2),
            ("phif0_kk_BW_BW", onp.array([11, 12]), 2),
            ("phif2_kk_BW_flatte1270", onp.array([17, 18, 19, 20, 21]), 5),
            ("phif2_kk_BW_BW", onp.array([33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]), 5),
        ]):
            for n, i in enumerate(range(0, len(const_index), num_iamps)):
                _args_list = copy.deepcopy(args_list)
                _const_index = const_index[i:i + num_iamps]
                leftover = [x for x in all_const_index if x not in _const_index]
                for idx in leftover:
                    _args_list[idx] = -100.0
                args_float_mod = onp.array(_args_list[float_list])
                wt = onp.asarray(pwaf.jit_weight_kk(args_float_mod))
                frac = onp.sum(wt) / sum_wt
                frac_list.append(float(frac))
                logger.info("Fraction {}: {:.6f}".format(mod_name + "_mod" + str(n), frac))
        logger.info("Fractions: {}".format(frac_list))

        ROOT.gROOT.SetBatch(True)
        c = ROOT.TCanvas("c_kk", "KK Fit Results", 1200, 800)
        c.Divide(2, 2)

        n_bins = 60
        f_min = 1.0
        f_max = 2.4
        phi_min = 1.01
        phi_max = 1.05

        data_f = onp.load("data/real_data/f_kk.npy").flatten()
        data_phi = onp.load("data/real_data/phi_kk.npy").flatten()

        c.cd(1)
        h_data_f = ROOT.TH1F("h_data_f", "f_{KK} Distribution; f_{KK} [GeV/c^{2}]; Events / bin", n_bins, f_min, f_max)
        for val in data_f:
            h_data_f.Fill(float(val))
        h_data_f.SetMarkerStyle(20)
        h_data_f.SetMarkerSize(0.6)
        h_data_f.Draw("E1")

        c.cd(2)
        h_data_phi = ROOT.TH1F("h_data_phi", "#phi_{KK} Distribution; #phi_{KK} [GeV/c^{2}]; Events / bin", n_bins, phi_min, phi_max)
        for val in data_phi:
            h_data_phi.Fill(float(val))
        h_data_phi.SetMarkerStyle(20)
        h_data_phi.SetMarkerSize(0.6)
        h_data_phi.Draw("E1")

        c.cd(3)
        mc_f = onp.load("data/mc_truth/f_kk.npy").flatten()
        h_mc_f = ROOT.TH1F("h_mc_f", "MC f_{KK} Distribution; f_{KK} [GeV/c^{2}]; Events / bin", n_bins, f_min, f_max)
        for val in mc_f:
            h_mc_f.Fill(float(val))
        h_mc_f.SetLineColor(ROOT.kRed)
        h_mc_f.Draw("HIST")

        c.cd(4)
        mc_phi = onp.load("data/mc_truth/phi_kk.npy").flatten()
        h_mc_phi = ROOT.TH1F("h_mc_phi", "MC #phi_{KK} Distribution; #phi_{KK} [GeV/c^{2}]; Events / bin", n_bins, phi_min, phi_max)
        for val in mc_phi:
            h_mc_phi.Fill(float(val))
        h_mc_phi.SetLineColor(ROOT.kRed)
        h_mc_phi.Draw("HIST")

        c.SaveAs("output/plot/kk_fit_plots.pdf")
        c.SaveAs("output/plot/kk_fit_plots.png")
        logger.info("Saved plots to output/plot/kk_fit_plots.pdf and .png")

        process_returns.set("parameter_values", args_list)
        process_returns.set("parameter_errors", onp.zeros_like(args_list))
        process_returns.set("min_fcn", likelihood)
        process_returns.info()

    def run_multiprocess(self):
        process_generator = Process_Initializer_Generator(**self.data_config)
        process_list = process_generator.process_initializer_generator()
        gpu_allocations = {}
        for gpu_idx in range(self.processes_gpus):
            assigned_gpu = self.total_gpu_id[gpu_idx % len(self.total_gpu_id)]
            gpu_allocations[gpu_idx] = assigned_gpu
        process_returns_list = []
        for idx, process_initializer in enumerate(process_list):
            if idx >= self.max_processes:
                break
            gpu_id = gpu_allocations[idx % self.processes_gpus]
            process_initializer.gpu_id = idx % self.thread_gpus
            process_initializer.total_gpu_id = [gpu_id]
            process_initializer.gpu_memory_limit_percentage = self.max_processes_memory
            process_returns = ProcessReturns()
            self.run([[process_initializer]], process_returns)
            process_returns_list.append(process_returns)
        for pr in process_returns_list:
            logger.info("Process result: {}".format(pr.get("min_fcn")))
        logger.info("All processes completed")


if __name__ == "__main__":
    logger = logging.getLogger("dplot")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    a = args()
    a.max_processes = 1
    a.max_processes_memory = 0.89
    a.processes_gpus = 2
    a.thread_gpus = 1
    a.threads_in_one_gpu = 1
    a.total_gpu_id = [0, 1]
    a.data_slices = 1
    a.mc_slices = 1
    a.mini_run = 1
    cl = Control(a)
    cl.run_multiprocess()