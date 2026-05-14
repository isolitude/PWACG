import os, sys, json, logging, logging.config
import jax.numpy as np
import numpy as onp
from dlib import dplex
from jax import device_put, jit, vmap, grad, value_and_grad, jvp
from functools import partial
from multiprocessing import Array, Barrier, Lock, Manager, Pipe, Process, Value
from threading import Thread
import time, glob, re, copy
from scipy.optimize import minimize
from iminuit import minimize as i_minimize
import pynvml

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

class Logger():
    def __init__(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_tensor.json", "r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)

logger = Logger("tensor")

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
        self.mc_phi_kk = None
        self.truth_phi_kk = None
        self.data_f_kk = None
        self.mc_f_kk = None
        self.truth_f_kk = None
        self.data_phif0_kk = None
        self.mc_phif0_kk = None
        self.truth_phif0_kk = None
        self.data_phif2_kk = None
        self.mc_phif2_kk = None
        self.truth_phif2_kk = None
        self.data_numbering = None
        self.gpu_id = None
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
        self.all_data_phi_kk = onp.array_split(self.data_phi_kk, num, axis=0)
        self.data_f_kk = onp.load("data/real_data/f_kk.npy")
        self.all_data_f_kk = onp.array_split(self.data_f_kk, num, axis=0)
        self.data_phif0_kk = onp.load("data/real_data/phif0_kk.npy")
        self.all_data_phif0_kk = onp.array_split(self.data_phif0_kk, num, axis=1)
        self.data_phif2_kk = onp.load("data/real_data/phif2_kk.npy")
        self.all_data_phif2_kk = onp.array_split(self.data_phif2_kk, num, axis=1)

    def mc_npz(self, num):
        self.mc_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_mc_phi_kk = onp.array_split(self.mc_phi_kk, num, axis=0)
        self.mc_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_mc_f_kk = onp.array_split(self.mc_f_kk, num, axis=0)
        self.mc_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_mc_phif0_kk = onp.array_split(self.mc_phif0_kk, num, axis=1)
        self.mc_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_mc_phif2_kk = onp.array_split(self.mc_phif2_kk, num, axis=1)

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

    def BW(self, m_, w_, Sbc):
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

    def phase(self, theta): return vmap(self._phase)(theta)
    def _phase(self, theta): return dplex.dconstruct(np.cos(theta), np.sin(theta))

    def jit_request(self):
        pass

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
        fvalue = [0]
        ferror = [0]
        min_fcn = 0
        stop_time = time.time()
        process_returns.set("parameter_values", fvalue)
        process_returns.set("parameter_errors", ferror)
        process_returns.set("min_fcn", min_fcn)
        process_returns.info()
        process_returns.set("timer", stop_time - start_time)
        process_returns.set("process_id", os.getpid())
        process_returns.set("gpu_id", process_initializer[0][0].total_gpu_id)

    def run_multiprocess(self):
        process_initializer_generator = Process_Initializer_Generator(
            data_slices=self.data_slices, mc_slices=self.mc_slices, mini_run=self.mini_run)
        process_initializer_generator.data_slices = self.data_slices
        process_initializer_generator.mc_slices = self.mc_slices
        process_initializer_generator.mini_run = self.mini_run
        process_initializer_generator.data_npz(self.data_slices)
        process_initializer_generator.mc_npz(self.mc_slices)
        process_initializer_generator.regular()
        process_initializer_list = list()
        for _ in process_initializer_generator.process_initializer_generator():
            process_initializer_list.append(_)
        process_pool = [Process()] * self.max_processes
        process_returns_list = [ProcessReturns() for _ in range(self.max_processes)]
        for i in range(self.max_processes):
            process_pool[i] = Process(target=self.run, args=([process_initializer_list[i]], process_returns_list[i]))
            process_pool[i].start()
        for i in range(self.max_processes):
            process_pool[i].join()
        for i in range(self.max_processes):
            process_returns_list[i].info()

if __name__ == '__main__':
    _args = args()
    _args.data_slices = 1
    _args.mc_slices = 1
    _args.mini_run = 1
    _args.max_processes = 1
    _args.thread_gpus = 1
    _args.threads_in_one_gpu = 1
    _args.gpu_memory_limit_percentage = 0.5
    _args.total_gpu_id = [0]
    cl = Control(_args)
    cl.run_multiprocess()