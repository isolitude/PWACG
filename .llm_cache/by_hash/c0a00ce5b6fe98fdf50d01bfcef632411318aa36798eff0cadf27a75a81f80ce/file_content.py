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


class args(object):
    def __init__(self):
        self.data_slices = None
        self.mc_slices = None
        self.mini_run = None
        self.max_processes = None
        self.thread_gpus = None
        self.threads_in_one_gpu = None
        self.gpu_memory_limit_percentage = None
        self.total_gpu_id = None
        self.data_numbering = None
        self.gpu_id = None
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
    def __init__(self, data_slices=None, mc_slices=None, mini_run=None, max_processes=None, thread_gpus=None, threads_in_one_gpu=None, gpu_memory_limit_percentage=None, total_gpu_id=None):
        self.data_slices = data_slices
        self.mc_slices = mc_slices
        self.mini_run = mini_run
        self.max_processes = max_processes
        self.thread_gpus = thread_gpus
        self.threads_in_one_gpu = threads_in_one_gpu
        self.gpu_memory_limit_percentage = gpu_memory_limit_percentage
        self.total_gpu_id = total_gpu_id

    def reader_amp(self, file_name):
        amp_list = list()
        with onp.load(file_name) as amp:
            for amp_name in amp.files:
                amp_list.append((amp[amp_name])[:,0:2])
        return onp.array(amp_list)

    def data_npz(self, num):
        self.data_phif0_kk = onp.load("data/real_data/phif0_kk.npy")
        self.all_data_phif0_kk = onp.array_split(self.data_phif0_kk, num, axis=1)
        self.data_phif2_kk = onp.load("data/real_data/phif2_kk.npy")
        self.all_data_phif2_kk = onp.array_split(self.data_phif2_kk, num, axis=1)
        data_phi_kk = onp.load("data/real_data/phi_kk.npy")
        self.all_data_phi_kk = onp.array_split(data_phi_kk, num, axis=0)
        data_f_kk = onp.load("data/real_data/f_kk.npy")
        self.all_data_f_kk = onp.array_split(data_f_kk, num, axis=0)
        wt_data_kk = onp.load("data/weight/weight_kk.npy")
        self.all_wt_data_kk = onp.array_split(wt_data_kk, num, axis=0)

    def mc_npz(self, num):
        self.mc_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_mc_phif0_kk = onp.array_split(self.mc_phif0_kk, num, axis=1)
        self.mc_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_mc_phif2_kk = onp.array_split(self.mc_phif2_kk, num, axis=1)
        mc_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_mc_phi_kk = onp.array_split(mc_phi_kk, num, axis=0)
        mc_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_mc_f_kk = onp.array_split(mc_f_kk, num, axis=0)

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
            _.data_phi_kk = self.all_data_phi_kk[data_numbering % self.data_slices]
            _.mc_phi_kk = self.all_mc_phi_kk[data_numbering % self.mc_slices]
            _.data_f_kk = self.all_data_f_kk[data_numbering % self.data_slices]
            _.mc_f_kk = self.all_mc_f_kk[data_numbering % self.mc_slices]
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
        self.truth_phi_kk = device_put(np.array(cdl.truth_phi_kk), device=self.device)
        self.truth_f_kk = device_put(np.array(cdl.truth_f_kk), device=self.device)
        self.truth_phif0_kk = device_put(np.array(cdl.truth_phif0_kk), device=self.device)
        self.truth_phif2_kk = device_put(np.array(cdl.truth_phif2_kk), device=self.device)
        self.wt_data_kk = device_put(np.array(cdl.wt_data_kk), device=self.device)

    def data_likelihood_kk(self, args):
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1,args[3]]).reshape(-1,2)
        kk_f980_theta = np.array([0.1,args[4]]).reshape(-1,2)
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7],args[8]]).reshape(-1,2)
        kk_f0_theta = np.array([args[9],args[10]]).reshape(-1,2)
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13],args[14],args[15],args[16],args[17]]).reshape(-1,5)
        kk_f1270_theta = np.array([args[18],args[19],args[20],args[21],args[22]]).reshape(-1,5)
        kk_f2_mass = np.array([args[23],args[24],args[25]])
        kk_f2_width = np.array([args[26],args[27],args[28]])
        kk_f2_const = np.array([args[29],args[30],args[31],args[32],args[33],args[34],args[35],args[36],args[37],args[38],args[39],args[40],args[41],args[42],args[43]]).reshape(-1,5)
        kk_f2_theta = np.array([args[44],args[45],args[46],args[47],args[48],args[49],args[50],args[51],args[52],args[53],args[54],args[55],args[56],args[57],args[58]]).reshape(-1,5)
        data_phif0_kk_BW_flatte980 = self.calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        data_phif0_kk_BW_BW = self.calculate_BW_BW(phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        data_phif2_kk_BW_flatte1270 = self.calculate_BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        data_phif2_kk_BW_BW = self.calculate_BW_BW(phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        lasso_data_phif0_kk_BW_flatte980 = self.lasso_calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        lasso_data_phif0_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        lasso_data_phif2_kk_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        lasso_data_phif2_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        sum_frac = np.sum(dplex.dabs(np.einsum("mljk->mjk", lasso_data_phif0_kk_BW_flatte980)+np.einsum("mljk->mjk", lasso_data_phif0_kk_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif2_kk_BW_flatte1270)+np.einsum("mljk->mjk", lasso_data_phif2_kk_BW_BW)))
        step_function = (np.power(np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_BW_flatte980))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_BW_flatte1270))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_BW_BW))/sum_frac)-1.03,2.0))*0.0 + 0.0
        return -np.sum(np.log(np.sum(dplex.dabs(data_phif0_kk_BW_flatte980 + data_phif0_kk_BW_BW + data_phif2_kk_BW_flatte1270 + data_phif2_kk_BW_BW),axis=1))) + 100.0*step_function

    def mc_likelihood_kk(self, args):
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1,args[3]]).reshape(-1,2)
        kk_f980_theta = np.array([0.1,args[4]]).reshape(-1,2)
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7],args[8]]).reshape(-1,2)
        kk_f0_theta = np.array([args[9],args[10]]).reshape(-1,2)
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13],args[14],args[15],args[16],args[17]]).reshape(-1,5)
        kk_f1270_theta = np.array([args[18],args[19],args[20],args[21],args[22]]).reshape(-1,5)
        kk_f2_mass = np.array([args[23],args[24],args[25]])
        kk_f2_width = np.array([args[26],args[27],args[28]])
        kk_f2_const = np.array([args[29],args[30],args[31],args[32],args[33],args[34],args[35],args[36],args[37],args[38],args[39],args[40],args[41],args[42],args[43]]).reshape(-1,5)
        kk_f2_theta = np.array([args[44],args[45],args[46],args[47],args[48],args[49],args[50],args[51],args[52],args[53],args[54],args[55],args[56],args[57],args[58]]).reshape(-1,5)
        mc_phif0_kk_BW_flatte980 = self.calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta, self.mc_phi_kk, self.mc_f_kk, self.mc_phif0_kk)
        mc_phif0_kk_BW_BW = self.calculate_BW_BW(phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta, self.mc_phi_kk, self.mc_f_kk, self.mc_phif0_kk)
        mc_phif2_kk_BW_flatte1270 = self.calculate_BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta, self.mc_phi_kk, self.mc_f_kk, self.mc_phif2_kk)
        mc_phif2_kk_BW_BW = self.calculate_BW_BW(phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta, self.mc_phi_kk, self.mc_f_kk, self.mc_phif2_kk)
        return np.sum(dplex.dabs(mc_phif0_kk_BW_flatte980 + mc_phif0_kk_BW_BW + mc_phif2_kk_BW_flatte1270 + mc_phif2_kk_BW_BW))

    def lasso_data_likelihood_kk(self, args):
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1,args[3]]).reshape(-1,2)
        kk_f980_theta = np.array([0.1,args[4]]).reshape(-1,2)
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7],args[8]]).reshape(-1,2)
        kk_f0_theta = np.array([args[9],args[10]]).reshape(-1,2)
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13],args[14],args[15],args[16],args[17]]).reshape(-1,5)
        kk_f1270_theta = np.array([args[18],args[19],args[20],args[21],args[22]]).reshape(-1,5)
        kk_f2_mass = np.array([args[23],args[24],args[25]])
        kk_f2_width = np.array([args[26],args[27],args[28]])
        kk_f2_const = np.array([args[29],args[30],args[31],args[32],args[33],args[34],args[35],args[36],args[37],args[38],args[39],args[40],args[41],args[42],args[43]]).reshape(-1,5)
        kk_f2_theta = np.array([args[44],args[45],args[46],args[47],args[48],args[49],args[50],args[51],args[52],args[53],args[54],args[55],args[56],args[57],args[58]]).reshape(-1,5)
        data_phif0_kk_BW_flatte980 = self.calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        data_phif0_kk_BW_BW = self.calculate_BW_BW(phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        data_phif2_kk_BW_flatte1270 = self.calculate_BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        data_phif2_kk_BW_BW = self.calculate_BW_BW(phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        lasso_data_phif0_kk_BW_flatte980 = self.lasso_calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        lasso_data_phif0_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        lasso_data_phif2_kk_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        lasso_data_phif2_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        sum_frac = np.sum(dplex.dabs(np.einsum("mljk->mjk", lasso_data_phif0_kk_BW_flatte980)+np.einsum("mljk->mjk", lasso_data_phif0_kk_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif2_kk_BW_flatte1270)+np.einsum("mljk->mjk", lasso_data_phif2_kk_BW_BW)))
        step_function = (np.power(np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_BW_flatte980))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_BW_flatte1270))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_BW_BW))/sum_frac)-1.03,2.0))*0.0 + 0.0
        return -np.sum(np.log(np.sum(dplex.dabs(data_phif0_kk_BW_flatte980 + data_phif0_kk_BW_BW + data_phif2_kk_BW_flatte1270 + data_phif2_kk_BW_BW),axis=1))) + np.power(10,0.0)*(np.sum(np.sqrt(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_BW_flatte980)))) + np.sum(np.sqrt(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_BW_BW)))) + np.sum(np.sqrt(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_BW_flatte1270)))) + np.sum(np.sqrt(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_BW_BW))))) + 100.0*step_function

    def weight_kk(self, args):
        phi_mass = np.array([1.02])
        phi_width = np.array([0.004])
        kk_f980_mass = np.array([args[0]])
        kk_f980_g_kk = np.array([args[1]])
        kk_f980_rg = np.array([args[2]])
        kk_f980_const = np.array([0.1,args[3]]).reshape(-1,2)
        kk_f980_theta = np.array([0.1,args[4]]).reshape(-1,2)
        kk_f0_mass = np.array([args[5]])
        kk_f0_width = np.array([args[6]])
        kk_f0_const = np.array([args[7],args[8]]).reshape(-1,2)
        kk_f0_theta = np.array([args[9],args[10]]).reshape(-1,2)
        kk_f1270_mass = np.array([args[11]])
        kk_f1270_width = np.array([args[12]])
        kk_f1270_const = np.array([args[13],args[14],args[15],args[16],args[17]]).reshape(-1,5)
        kk_f1270_theta = np.array([args[18],args[19],args[20],args[21],args[22]]).reshape(-1,5)
        kk_f2_mass = np.array([args[23],args[24],args[25]])
        kk_f2_width = np.array([args[26],args[27],args[28]])
        kk_f2_const = np.array([args[29],args[30],args[31],args[32],args[33],args[34],args[35],args[36],args[37],args[38],args[39],args[40],args[41],args[42],args[43]]).reshape(-1,5)
        kk_f2_theta = np.array([args[44],args[45],args[46],args[47],args[48],args[49],args[50],args[51],args[52],args[53],args[54],args[55],args[56],args[57],args[58]]).reshape(-1,5)
        data_phif0_kk_BW_flatte980 = self.calculate_BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        data_phif0_kk_BW_BW = self.calculate_BW_BW(phi_mass, phi_width, kk_f0_mass, kk_f0_width, kk_f0_const, kk_f0_theta, self.data_phi_kk, self.data_f_kk, self.data_phif0_kk)
        data_phif2_kk_BW_flatte1270 = self.calculate_BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        data_phif2_kk_BW_BW = self.calculate_BW_BW(phi_mass, phi_width, kk_f2_mass, kk_f2_width, kk_f2_const, kk_f2_theta, self.data_phi_kk, self.data_f_kk, self.data_phif2_kk)
        return np.sum(dplex.dabs(data_phif0_kk_BW_flatte980 + data_phif0_kk_BW_BW + data_phif2_kk_BW_flatte1270 + data_phif2_kk_BW_BW),axis=1)

    def calculate_BW_flatte980(self, phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta, sbc_phi, sbc_f, amp):
        bw = self.BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def lasso_calculate_BW_flatte980(self, phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta, sbc_phi, sbc_f, amp):
        bw = self.BW_flatte980(phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        return phif

    def calculate_BW_BW(self, phi_mass, phi_width, kk_f_mass, kk_f_width, kk_f_const, kk_f_theta, sbc_phi, sbc_f, amp):
        bw = self.BW_BW(phi_mass, phi_width, kk_f_mass, kk_f_width, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(kk_f_const, kk_f_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def lasso_calculate_BW_BW(self, phi_mass, phi_width, kk_f_mass, kk_f_width, kk_f_const, kk_f_theta, sbc_phi, sbc_f, amp):
        bw = self.BW_BW(phi_mass, phi_width, kk_f_mass, kk_f_width, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(kk_f_const, kk_f_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        return phif

    def calculate_BW_flatte1270(self, phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta, sbc_phi, sbc_f, amp):
        bw = self.BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(kk_f1270_const, kk_f1270_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def lasso_calculate_BW_flatte1270(self, phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta, sbc_phi, sbc_f, amp):
        bw = self.BW_flatte1270(phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, sbc_phi, sbc_f)
        const_ph = dplex.dconstruct(kk_f1270_const, kk_f1270_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        return phif

    def BW_flatte980(self, phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, sbc_phi, sbc_f):
        a = self.BW(phi_mass, phi_width, sbc_phi)
        b = vmap(partial(self.flatte980, Sbc=sbc_f), out_axes=1)(kk_f980_mass, kk_f980_g_kk, kk_f980_rg)
        return dplex.deinsum("j,ij->ij", a, b)

    def BW_BW(self, phi_mass, phi_width, kk_f_mass, kk_f_width, sbc_phi, sbc_f):
        a = self.BW(phi_mass, phi_width, sbc_phi)
        b = vmap(partial(self.BW_fside, Sbc=sbc_f), out_axes=1)(kk_f_mass, kk_f_width)
        return dplex.deinsum("j,ij->ij", a, b)

    def BW_flatte1270(self, phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, sbc_phi, sbc_f):
        a = self.BW(phi_mass, phi_width, sbc_phi)
        b = vmap(partial(self.flatte1270, Sbc=sbc_f), out_axes=1)(kk_f1270_mass, kk_f1270_width)
        return dplex.deinsum("j,ij->ij", a, b)

    def BW(self, m_, w_, Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc, -m_*w_*np.ones(l))
        return dplex.ddivide(1.0, temp)

    def BW_fside(self, m_, w_, Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc, -m_*w_*np.ones(l))
        return dplex.ddivide(1.0, temp)

    def BW_relativity(self, m_, w_, Sbc):
        gamma = np.sqrt(m_*m_*(m_*m_+w_*w_))
        k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc, -m_*w_*np.ones(l))
        return dplex.ddivide(k, temp)

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

    def flatte500(self, m_, b1, b2, b3, b4, b5, Sbc):
        m2 = m_*m_; rp = 0.139556995; mpi2d2 = 0.009739946882
        cro1 = np.sqrt(np.abs((Sbc-(2*rp)**2)*Sbc))/Sbc
        cro2 = np.sqrt(np.abs((m2-(2*rp)**2)*m2))/m2
        pip1 = np.sqrt(np.abs(1.0 - 0.3116765584/Sbc))/(1.0+np.exp(9.8-3.5*Sbc))
        pip2 = np.sqrt(np.abs(1.0 - 0.3116765584/m2))/(1.0+np.exp(9.8-3.5*m2))
        cgam1 = m_*(b1+b2*Sbc)*(Sbc-mpi2d2)/(m2-mpi2d2)*np.exp(-(Sbc-m2)/b3)*cro1/cro2
        cgam2 = m_*b4*pip1/pip2
        tmp = dplex.dconstruct(m2-Sbc, -b5*(cgam1+cgam2))
        return dplex.ddivide(1.0, tmp)

    def hvp_data_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.data_likelihood_kk), [args_float], [any_vector])
    def hvp_mc_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.mc_likelihood_kk), [args_float], [any_vector])

    def phase(self, theta): return vmap(self._phase)(theta)
    def _phase(self, theta): return dplex.dconstruct(np.cos(theta), np.sin(theta))

    def jit_request(self):
        self.jit_data_likelihood_kk = jit(self.data_likelihood_kk, device=self.device)
        self.jit_mc_likelihood_kk = jit(self.mc_likelihood_kk, device=self.device)
        self.jit_lasso_data_likelihood_kk = jit(self.lasso_data_likelihood_kk, device=self.device)
        self.jit_grad_data_likelihood_kk = jit(grad(self.data_likelihood_kk), device=self.device)
        self.jit_grad_mc_likelihood_kk = jit(value_and_grad(self.mc_likelihood_kk), device=self.device)
        self.jit_hvp_data_fwdrev_kk = jit(self.hvp_data_fwdrev_kk, device=self.device)
        self.jit_hvp_mc_fwdrev_kk = jit(self.hvp_mc_fwdrev_kk, device=self.device)
        self.jit_weight_kk = jit(self.weight_kk, device=self.device)


class Control(object):
    def __init__(self, args):
        self.bic_delete_mods = args.bic_delete_mods
        self.data_slices = args.data_slices
        self.mc_slices = args.mc_slices
        self.mini_run = args.mini_run
        self.max_processes = args.max_processes
        self.thread_gpus = args.thread_gpus
        self.threads_in_one_gpu = args.threads_in_one_gpu
        self.gpu_memory_limit_percentage = args.gpu_memory_limit_percentage
        self.total_gpu_id = args.total_gpu_id
        self.data_config = dict()
        self.data_config["data_slices"] = args.data_slices
        self.data_config["mc_slices"] = args.mc_slices
        self.data_config["mini_run"] = args.mini_run
        self.data_config["max_processes"] = args.max_processes
        self.data_config["thread_gpus"] = args.thread_gpus
        self.data_config["threads_in_one_gpu"] = args.threads_in_one_gpu
        self.data_config["gpu_memory_limit_percentage"] = args.gpu_memory_limit_percentage
        self.data_config["total_gpu_id"] = args.total_gpu_id
        self.args_list = onp.array([1.02, 0.004, 0.98, 0.1, 0.1, 0.1, 1.704, 0.123, 0.1, 0.1, 0.1, 0.1, 1.2755, 0.1867, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.517, 2.157, 2.345, 0.086, 0.152, 0.322, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.float_list = onp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])

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
            self.kk_data_lh[device_rank, _] = self.pwaf_list[device_rank][_].jit_lasso_data_likelihood_kk(args_float)
            self.kk_mc_lh[device_rank, _] = self.pwaf_list[device_rank][_].jit_mc_likelihood_kk(args_float)

    def thread_likelihood(self, args_float):
        threading_list = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target=self.likelihood_in_sigle_device, args=(args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus): threading_list[_].start()
        for _ in range(self.thread_gpus): threading_list[_].join()
        result_kk = onp.sum(self.kk_data_lh) + 100.0*onp.log(onp.sum(self.kk_mc_lh)/1000.0)
        result = result_kk
        return result

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
        result_kk = onp.sum(self.kk_data_grad, axis=(0,1)) + 100.0*onp.sum(self.kk_mc_grad, axis=(0,1))/onp.sum(self.kk_mc_grad_v)
        result = result_kk
        return result

    def thread_hvp(self, args_float, any_vector):
        threading_list = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target=self.hvp_in_sigle_device, args=(args_float, any_vector, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus): threading_list[_].start()
        for _ in range(self.thread_gpus): threading_list[_].join()
        result_kk = onp.sum(self.kk_data_hvp, axis=(0,1)) + 100.0*(onp.sum(self.kk_mc_hvp, axis=(0,1))/onp.sum(self.kk_mc_hvp_v) - onp.sum(self.kk_mc_hvp_g, axis=(0,1))*onp.sum(self.kk_mc_hvp_v)/(onp.sum(self.kk_mc_hvp_v)**2))
        result = result_kk
        return result

    def hvp_in_sigle_device(self, args_float, any_vector, device_rank):
        for _ in range(self.threads_in_one_gpu):
            _, self.kk_data_hvp[device_rank, _, :] = self.pwaf_list[device_rank][_].jit_hvp_data_fwdrev_kk(args_float, any_vector)
            self.kk_mc_hvp_v[device_rank, _], self.kk_mc_hvp_g[device_rank, _, :], self.kk_mc_hvp[device_rank, _, :] = self.pwaf_list[device_rank][_].jit_hvp_mc_fwdrev_kk(args_float, any_vector)

    def compile_func(self):
        vector = onp.ones(self.args_float.shape)
        self.thread_likelihood(self.args_float)
        self.thread_grad_likelihood(self.args_float)
        self.thread_hvp(self.args_float, vector)

    def run(self, process_initializer, process_returns):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x:str(x), process_initializer[0][0].total_gpu_id))
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(process_initializer[0][0].gpu_memory_limit_percentage)
        config.update("jax_enable_x64", True)
        device_list = jdevices()
        self.pwaf_list = [[PWAFunc(process_initializer[i][j], device_list[process_initializer[i][j].gpu_id])
                          for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
        for i in range(self.thread_gpus):
            for j in range(self.threads_in_one_gpu):
                self.pwaf_list[i][j].jit_request()
        num_seed = onp.random.randint(low=0, high=1000000, size=1, dtype='l')
        logger.info("random seed: {}".format(num_seed))
        onp.random.seed(num_seed)
        self.args_float = self.args_list[self.float_list]
        self.compile_func()
        min_fcn = 0
        for n in range(1):
            args_float = self.args_float
            t1 = time.time()
            res = minimize(fun=self.thread_likelihood, x0=args_float,
                          jac=self.thread_grad_likelihood, hessp=self.thread_hvp,
                          method="Newton-CG", callback=self.my_callback,
                          options={"disp": False, "xtol": 1e-8})
            t2 = time.time()
            logger.info("iteration {}: fcn={}, time={}".format(n, res.fun, t2-t1))
            if res.fun < min_fcn or n == 0:
                min_fcn = res.fun
                fvalue = res.x
            self.save_in_json(fvalue, str(n))
        ferror = [min_fcn]
        process_returns.set("parameter_values", fvalue)
        process_returns.set("parameter_errors", ferror)
        process_returns.set("min_fcn", min_fcn)
        process_returns.info()
        process_returns.set("timer", t2 - t1)
        process_returns.set("process_id", os.getpid())
        process_returns.set("gpu_id", process_initializer[0][0].total_gpu_id)

    def my_callback(self, xk):
        pass

    def run_multiprocess(self):
        process_initializer_generator = Process_Initializer_Generator(
            data_slices=self.data_slices, mc_slices=self.mc_slices,
            mini_run=self.mini_run, max_processes=self.max_processes,
            thread_gpus=self.thread_gpus, threads_in_one_gpu=self.threads_in_one_gpu,
            gpu_memory_limit_percentage=self.gpu_memory_limit_percentage,
            total_gpu_id=self.total_gpu_id)
        process_returns_list = [ProcessReturns() for _ in range(self.max_processes)]
        process_initializer_list = [list() for _ in range(self.max_processes)]
        for i, process_initializer in enumerate(process_initializer_generator.process_initializer_generator()):
            process_initializer_list[i % self.max_processes].append(process_initializer)
        for i in range(self.max_processes):
            self.process_pool[i] = Process(target=self.run, args=(process_initializer_list[i], process_returns_list[i]))
            self.process_pool[i].start()
        for i in range(self.max_processes):
            self.process_pool[i].join()
        for i in range(self.max_processes):
            process_returns_list[i].info()

    def save_in_json(self, np_values, save_addr="", n="", result_info={}):
        import json
        pwa_info = {"parameters": {}}
        for idx, val in enumerate(np_values):
            pwa_info["parameters"][str(idx)] = float(val)
        with open("output/fit/fit_result_kk/pwa_info_{}.json".format(n), "w") as f:
            json.dump(pwa_info, f, indent=2)


if __name__ == '__main__':
    logger = logging.getLogger("lasso")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    args_obj = args()
    args_obj.data_slices = 1
    args_obj.mc_slices = 1
    args_obj.mini_run = 1
    args_obj.max_processes = 1
    args_obj.thread_gpus = 1
    args_obj.threads_in_one_gpu = 1
    args_obj.gpu_memory_limit_percentage = 0.5
    args_obj.total_gpu_id = [0]
    cl = Control(args_obj)
    cl.run_multiprocess()