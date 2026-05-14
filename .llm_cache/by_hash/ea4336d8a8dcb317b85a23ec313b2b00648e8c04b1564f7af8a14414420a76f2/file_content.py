

import copy
import json
import logging
import logging.config
import os
import sys
import re
import time
import glob
from functools import partial
from multiprocessing import Array, Barrier, Lock, Manager, Pipe, Process, Value
from threading import Thread

import jax.numpy as np
import numpy as onp
from dlib import dplex
from jax import device_put
from jax import devices as jdevices
from jax import grad, hessian, jit, jvp, value_and_grad, vmap
from jax import config
from scipy.optimize import minimize

from iminuit import minimize as i_minimize
import pynvml


logger = logging.getLogger("draw")


class args(object):
    def __init__(self):
        self.total_gpu_id = None
        self.processes_gpus = None
        self.max_processes = None
        self.max_processes_memory = None
        self.thread_gpus = None
        self.threads_in_one_gpu = None
        
        self.data_slices = None
        self.mc_slices = None
        self.mini_run = None
        
        self.bic_delete_mods = list()




class ProcessInitializers:
    def __init__(self):
        
        self.data_phif0_kk = None
        self.mc_phif0_kk = None
        self.truth_phif0_kk = None
        
        self.data_phif2_kk = None
        self.mc_phif2_kk = None
        self.truth_phif2_kk = None
        
        self.data_phi_kk = None
        self.mc_phi_kk = None
        self.truth_phi_kk = None
        
        self.data_f_kk = None
        self.mc_f_kk = None
        self.truth_f_kk = None
        
        
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
    

    def reader_amp(self,file_name):
        amp_list = list()
        with onp.load(file_name) as amp:
            for amp_name in amp.files:
                amp_list.append((amp[amp_name])[:,0:2])
        return onp.array(amp_list)

    def data_npz(self,num):
        
        self.data_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_data_phif0_kk = onp.array_split(self.data_phif0_kk,num,axis=1)
        
        self.data_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_data_phif2_kk = onp.array_split(self.data_phif2_kk,num,axis=1)
        
        
        data_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_data_f_kk = onp.array_split(data_f_kk,num,axis=0)
        
        data_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_data_phi_kk = onp.array_split(data_phi_kk,num,axis=0)
        
        data_b123_kk = onp.load("data/mc_truth/b123_kk.npy")
        self.all_data_b123_kk = onp.array_split(data_b123_kk,num,axis=0)
        
        data_b124_kk = onp.load("data/mc_truth/b124_kk.npy")
        self.all_data_b124_kk = onp.array_split(data_b124_kk,num,axis=0)
        
        
        wt_data_kk = onp.load("data/weight/weight_kk.npy")
        self.all_wt_data_kk = onp.array_split(wt_data_kk,num,axis=0)
        

    def mc_npz(self,num):
        
        self.mc_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_mc_phif0_kk = onp.array_split(self.mc_phif0_kk,num,axis=1)
        
        self.mc_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_mc_phif2_kk = onp.array_split(self.mc_phif2_kk,num,axis=1)
        
        
        mc_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_mc_f_kk = onp.array_split(mc_f_kk,num,axis=0)
        
        mc_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_mc_phi_kk = onp.array_split(mc_phi_kk,num,axis=0)
        
        mc_b123_kk = onp.load("data/mc_truth/b123_kk.npy")
        self.all_mc_b123_kk = onp.array_split(mc_b123_kk,num,axis=0)
        
        mc_b124_kk = onp.load("data/mc_truth/b124_kk.npy")
        self.all_mc_b124_kk = onp.array_split(mc_b124_kk,num,axis=0)
        

    def truth_npz(self,num):
        
        self.truth_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_truth_phif0_kk = self.truth_phif0_kk[:,0:150000]
        
        self.truth_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_truth_phif2_kk = self.truth_phif2_kk[:,0:150000]
        
        
        truth_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_truth_f_kk = truth_f_kk[0:150000]
        
        truth_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_truth_phi_kk = truth_phi_kk[0:150000]
        
        truth_b123_kk = onp.load("data/mc_truth/b123_kk.npy")
        self.all_truth_b123_kk = truth_b123_kk[0:150000]
        
        truth_b124_kk = onp.load("data/mc_truth/b124_kk.npy")
        self.all_truth_b124_kk = truth_b124_kk[0:150000]
        

    def regular(self):
        
        self.re_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        
        self.re_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        
        
        regular_phif0_kk = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif0_kk)**2,axis=2)),axis=1)
        self.all_data_phif0_kk = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_phif0_kk),regular_phif0_kk)
        self.all_mc_phif0_kk = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_phif0_kk),regular_phif0_kk)
        self.all_truth_phif0_kk = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_phif0_kk),regular_phif0_kk)
        
        regular_phif2_kk = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif2_kk)**2,axis=2)),axis=1)
        self.all_data_phif2_kk = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_phif2_kk),regular_phif2_kk)
        self.all_mc_phif2_kk = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_phif2_kk),regular_phif2_kk)
        self.all_truth_phif2_kk = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_phif2_kk),regular_phif2_kk)
        
    
    def process_initializer_generator(self):
        self.data_npz(self.data_slices)
        self.mc_npz(self.mc_slices)
        self.truth_npz(self.mc_slices)
        self.regular()
        logger.info("============ w i t h  n e x t ===============")

        data_numbering = 0
        event_num = self.mini_run
        while data_numbering<event_num:
            _ = ProcessInitializers()
            # 这里 %self.data_slices 是为了在多进程的时候使程序只靠 mini_run 控制循环次数
            
            _.data_f_kk = self.all_data_f_kk[data_numbering%self.data_slices]
            _.mc_f_kk = self.all_mc_f_kk[data_numbering%self.mc_slices]
            _.truth_f_kk = self.all_truth_f_kk
            
            _.data_phi_kk = self.all_data_phi_kk[data_numbering%self.data_slices]
            _.mc_phi_kk = self.all_mc_phi_kk[data_numbering%self.mc_slices]
            _.truth_phi_kk = self.all_truth_phi_kk
            
            _.data_b123_kk = self.all_data_b123_kk[data_numbering%self.data_slices]
            _.mc_b123_kk = self.all_mc_b123_kk[data_numbering%self.mc_slices]
            _.truth_b123_kk = self.all_truth_b123_kk
            
            _.data_b124_kk = self.all_data_b124_kk[data_numbering%self.data_slices]
            _.mc_b124_kk = self.all_mc_b124_kk[data_numbering%self.mc_slices]
            _.truth_b124_kk = self.all_truth_b124_kk
            
            
            _.data_phif0_kk = self.all_data_phif0_kk[data_numbering%self.data_slices]
            _.mc_phif0_kk = self.all_mc_phif0_kk[data_numbering%self.mc_slices]
            _.truth_phif0_kk = self.all_truth_phif0_kk
            
            _.data_phif2_kk = self.all_data_phif2_kk[data_numbering%self.data_slices]
            _.mc_phif2_kk = self.all_mc_phif2_kk[data_numbering%self.mc_slices]
            _.truth_phif2_kk = self.all_truth_phif2_kk
            
            
            _.wt_data_kk = self.all_wt_data_kk[data_numbering%self.data_slices]
            
            _.data_numbering = data_numbering
            yield _
            data_numbering += 1
        return

        




class ProcessReturns:
    def __init__(self):
        self.manager = Manager()
        self._dict = self.manager.dict(
            data_numbering=None,
            process_id=None,
            parameter_values=None,
            parameter_errors=None,
            min_fcn=None,
            timer=None, 
            gpu_id=None)
    def set(self, key, value):
        self._dict[key] = value

    def get(self, key):
        return self._dict[key]

    def info(self):
        logger.info("No. {} data on GPU{} for {} s".format(
            self._dict["data_numbering"],
            self._dict["gpu_id"],
            self._dict["timer"]))
        for value, error in zip(self._dict["parameter_values"], self._dict["parameter_errors"]):
            logger.debug("value={}, error={}".format(value, error))






class PWAFunc():
    def __init__(self, cdl=None, device_id=None):
        self.device = device_id
        
        self.data_phif0_kk = device_put(np.array(cdl.data_phif0_kk),device=self.device)
        self.mc_phif0_kk = device_put(np.array(cdl.mc_phif0_kk),device=self.device)
        self.truth_phif0_kk = device_put(np.array(cdl.truth_phif0_kk),device=self.device)
        
        self.data_phif2_kk = device_put(np.array(cdl.data_phif2_kk),device=self.device)
        self.mc_phif2_kk = device_put(np.array(cdl.mc_phif2_kk),device=self.device)
        self.truth_phif2_kk = device_put(np.array(cdl.truth_phif2_kk),device=self.device)
        
        self.data_phi_kk = device_put(np.array(cdl.data_phi_kk),device=self.device)
        self.mc_phi_kk = device_put(np.array(cdl.mc_phi_kk),device=self.device)
        self.truth_phi_kk = device_put(np.array(cdl.truth_phi_kk),device=self.device)
        
        self.data_f_kk = device_put(np.array(cdl.data_f_kk),device=self.device)
        self.mc_f_kk = device_put(np.array(cdl.mc_f_kk),device=self.device)
        self.truth_f_kk = device_put(np.array(cdl.truth_f_kk),device=self.device)
        
        
        self.wt_data_kk = device_put(np.array(cdl.wt_data_kk),device=self.device)
        
    
    
        
    
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
        
        
        data_phif0_kk_BW_flatte980 = self.calculate_BW_flatte980(phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const,kk_f980_theta,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk)
        
        data_phif0_kk_BW_BW = self.calculate_BW_BW(phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const,kk_f0_theta,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk)
        
        data_phif2_kk_BW_flatte1270 = self.calculate_BW_flatte1270(phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const,kk_f1270_theta,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk)
        
        data_phif2_kk_BW_BW = self.calculate_BW_BW(phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const,kk_f2_theta,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk)
        
        
        lasso_data_phif0_kk_BW_flatte980 = self.lasso_calculate_BW_flatte980(phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const,kk_f980_theta,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk)
        
        lasso_data_phif0_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const,kk_f0_theta,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk)
        
        lasso_data_phif2_kk_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const,kk_f1270_theta,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk)
        
        lasso_data_phif2_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const,kk_f2_theta,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk)
        
        step_function = 0.0
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
        
        
        mc_phif0_kk_BW_flatte980 = self.calculate_BW_flatte980(phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const,kk_f980_theta,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk)
        
        mc_phif0_kk_BW_BW = self.calculate_BW_BW(phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const,kk_f0_theta,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk)
        
        mc_phif2_kk_BW_flatte1270 = self.calculate_BW_flatte1270(phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const,kk_f1270_theta,self.mc_phi_kk,self.mc_f_kk,self.mc_phif2_kk)
        
        mc_phif2_kk_BW_BW = self.calculate_BW_BW(phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const,kk_f2_theta,self.mc_phi_kk,self.mc_f_kk,self.mc_phif2_kk)
        
        return np.sum(dplex.dabs(mc_phif0_kk_BW_flatte980 + mc_phif0_kk_BW_BW + mc_phif2_kk_BW_flatte1270 + mc_phif2_kk_BW_BW))
    



    
        
    
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
        
        
        lasso_data_phif0_kk_BW_flatte980 = self.lasso_calculate_BW_flatte980(phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const,kk_f980_theta,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk)
        
        lasso_data_phif0_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const,kk_f0_theta,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk)
        
        lasso_data_phif2_kk_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const,kk_f1270_theta,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk)
        
        lasso_data_phif2_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const,kk_f2_theta,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk)
        
        total_wt = dplex.dabs(np.einsum("ljk->jk", lasso_data_phif0_kk_BW_flatte980)+np.einsum("ljk->jk", lasso_data_phif0_kk_BW_BW)+np.einsum("ljk->jk", lasso_data_phif2_kk_BW_flatte1270)+np.einsum("ljk->jk", lasso_data_phif2_kk_BW_BW))
        wt_list = [total_wt,dplex.dabs(lasso_data_phif0_kk_BW_flatte980),dplex.dabs(lasso_data_phif0_kk_BW_BW),dplex.dabs(lasso_data_phif2_kk_BW_flatte1270),dplex.dabs(lasso_data_phif2_kk_BW_BW)]
        return wt_list
    

        
    
    def weight_truth_kk(self, args):
        
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
        
        
        lasso_data_phif0_kk_BW_flatte980 = self.lasso_calculate_BW_flatte980(phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const,kk_f980_theta,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk)
        
        lasso_data_phif0_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const,kk_f0_theta,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk)
        
        lasso_data_phif2_kk_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const,kk_f1270_theta,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk)
        
        lasso_data_phif2_kk_BW_BW = self.lasso_calculate_BW_BW(phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const,kk_f2_theta,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk)
        
        total_wt = dplex.dabs(np.einsum("ljk->jk", lasso_data_phif0_kk_BW_flatte980)+np.einsum("ljk->jk", lasso_data_phif0_kk_BW_BW)+np.einsum("ljk->jk", lasso_data_phif2_kk_BW_flatte1270)+np.einsum("ljk->jk", lasso_data_phif2_kk_BW_BW))
        wt_list = [total_wt,dplex.dabs(lasso_data_phif0_kk_BW_flatte980),dplex.dabs(lasso_data_phif0_kk_BW_BW),dplex.dabs(lasso_data_phif2_kk_BW_flatte1270),dplex.dabs(lasso_data_phif2_kk_BW_BW)]
        return wt_list
    


    
    def calculate_BW_flatte980(self,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const,kk_f980_theta,phi_kk,f_kk,phif0_kk):

        # ph = np.moveaxis(self.phase(kk_f980_theta), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte980(phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f980_const), ph)
        # print("const_ph",const_ph.shape)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif 

    def BW_flatte980(self, phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        b = vmap(partial(self.flatte980, Sbc=f_kk), out_axes=1)(kk_f980_mass, kk_f980_g_kk, kk_f980_rg)
        return dplex.deinsum("j, ij->ij",a,b)

    def calculate_BW_BW(self,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const,kk_f0_theta,phi_kk,f_kk,phif0_kk):

        # ph = np.moveaxis(self.phase(kk_f0_theta), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW(phi_mass,phi_width,phi_kk,kk_f0_mass,kk_f0_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0_const, kk_f0_theta)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0_const), ph)
        # print("const_ph",const_ph.shape)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif 

    def BW_BW(self, phi_mass,phi_width,phi_kk,kk_f0_mass,kk_f0_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        b = vmap(partial(self.BW, Sbc=f_kk), out_axes=1)(kk_f0_mass, kk_f0_width)
        return dplex.deinsum("j, ij->ij",a,b)

    def calculate_BW_flatte1270(self,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const,kk_f1270_theta,phi_kk,f_kk,phif2_kk):

        # ph = np.moveaxis(self.phase(kk_f1270_theta), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte1270(phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f1270_const, kk_f1270_theta)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f1270_const), ph)
        # print("const_ph",const_ph.shape)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif 

    def BW_flatte1270(self, phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        b = vmap(partial(self.flatte1270, Sbc=f_kk), out_axes=1)(kk_f1270_mass, kk_f1270_width)
        return dplex.deinsum("j, ij->ij",a,b)
    
    def lasso_calculate_BW_flatte980(self,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const,kk_f980_theta,phi_kk,f_kk,phif0_kk):

        # ph = np.moveaxis(self.phase(kk_f980_theta), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte980(phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f980_const), ph)
        # print("const_ph",const_ph.shape)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        return phif
    
    def lasso_calculate_BW_BW(self,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const,kk_f0_theta,phi_kk,f_kk,phif0_kk):

        # ph = np.moveaxis(self.phase(kk_f0_theta), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW(phi_mass,phi_width,phi_kk,kk_f0_mass,kk_f0_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0_const, kk_f0_theta)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0_const), ph)
        # print("const_ph",const_ph.shape)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        return phif
    
    def lasso_calculate_BW_flatte1270(self,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const,kk_f1270_theta,phi_kk,f_kk,phif2_kk):

        # ph = np.moveaxis(self.phase(kk_f1270_theta), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte1270(phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f1270_const, kk_f1270_theta)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f1270_const), ph)
        # print("const_ph",const_ph.shape)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        return phif

    def phase(self, theta):
        return vmap(self._phase)(theta)

    def _phase(self, theta):
        return dplex.dconstruct(np.cos(theta), np.sin(theta))
    
    
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


    
    def hvp_data_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.data_likelihood_kk), [args_float], [any_vector])

    def hvp_mc_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.mc_likelihood_kk), [args_float], [any_vector])
    

    def jit_request(self):
        
        self.jit_weight_kk = jit(self.weight_kk, device=self.device)
        self.jit_weight_truth_kk = jit(self.weight_truth_kk, device=self.device)
        

        
        self.jit_data_likelihood_kk = jit(self.data_likelihood_kk, device=self.device)
        self.jit_mc_likelihood_kk = jit(self.mc_likelihood_kk, device=self.device)
        self.jit_grad_data_likelihood_kk = jit(grad(self.data_likelihood_kk), device=self.device)
        self.jit_grad_mc_likelihood_kk = jit(value_and_grad(self.mc_likelihood_kk), device=self.device)
        self.jit_hvp_data_fwdrev_kk = jit(self.hvp_data_fwdrev_kk, device=self.device)
        self.jit_hvp_mc_fwdrev_kk = jit(self.hvp_mc_fwdrev_kk, device=self.device)
        

    
    def mod_weight(self, total_weight, mod_name, args_result, const_index, num_iamps, all_const_index):
        for n, i in enumerate(range(0, len(const_index), num_iamps)):
            _args_result = copy.deepcopy(args_result)
            _const_index = const_index[i:i+num_iamps]
            leftover = [x for x in all_const_index if x not in _const_index]
            for i in leftover:
                _args_result[i] = -100.0 
            args_float = np.array(_args_result[self.float_list])
            wt = self.jit_weight_kk(args_float)
            # print(mod_name, wt)
            frac = onp.sum(wt)/self.sum_wt
            print(mod_name, frac)
            total_weight[mod_name+"_"+str(n)] = wt
        return

    def run_weight(self, args_list, float_list,mode="pass"):
        all_const_index = [5, 6, 11, 12, 17, 18, 19, 20, 21, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        self.args_list = args_list
        self.float_list = float_list
        print("run weight")
        self.jit_request()
        args_float = np.array(self.args_list[self.float_list])
        if mode == "pass":
            wt_list = self.jit_weight_kk(args_float)
        if mode == "truth":
            wt_list = self.jit_weight_truth_kk(args_float)
        self.sum_wt = onp.sum(wt_list[0])
        total_weight = dict()
        # print("args_list: \n",args_float)
        total_weight["all_mods_wt"] = wt_list[0]
        total_fit_frac = 0
        loop_index = 0
        temp_cal_cul_func = list()
        
        num_mod = int(len([5, 6])/2)
        for i in range(num_mod):
            if not "phif0_kk_BW_flatte980" in temp_cal_cul_func:
                loop_index += 1
            wt = wt_list[loop_index][i]
            total_weight["phif0_kk_BW_flatte980_"+str(i)] = wt
            print("phif0_kk_BW_flatte980 frac", onp.sum(wt)/self.sum_wt)
            total_fit_frac += onp.sum(wt)/self.sum_wt
            temp_cal_cul_func.append("phif0_kk_BW_flatte980")
        
        num_mod = int(len([11, 12])/2)
        for i in range(num_mod):
            if not "phif0_kk_BW_BW" in temp_cal_cul_func:
                loop_index += 1
            wt = wt_list[loop_index][i]
            total_weight["phif0_kk_BW_BW_"+str(i)] = wt
            print("phif0_kk_BW_BW frac", onp.sum(wt)/self.sum_wt)
            total_fit_frac += onp.sum(wt)/self.sum_wt
            temp_cal_cul_func.append("phif0_kk_BW_BW")
        
        num_mod = int(len([17, 18, 19, 20, 21])/5)
        for i in range(num_mod):
            if not "phif2_kk_BW_flatte1270" in temp_cal_cul_func:
                loop_index += 1
            wt = wt_list[loop_index][i]
            total_weight["phif2_kk_BW_flatte1270_"+str(i)] = wt
            print("phif2_kk_BW_flatte1270 frac", onp.sum(wt)/self.sum_wt)
            total_fit_frac += onp.sum(wt)/self.sum_wt
            temp_cal_cul_func.append("phif2_kk_BW_flatte1270")
        
        num_mod = int(len([33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47])/5)
        for i in range(num_mod):
            if not "phif2_kk_BW_BW" in temp_cal_cul_func:
                loop_index += 1
            wt = wt_list[loop_index][i]
            total_weight["phif2_kk_BW_BW_"+str(i)] = wt
            print("phif2_kk_BW_BW frac", onp.sum(wt)/self.sum_wt)
            total_fit_frac += onp.sum(wt)/self.sum_wt
            temp_cal_cul_func.append("phif2_kk_BW_BW")
        
        values = np.array(self.args_list)
        total_weight["fit_value"] = values
        ferror = ["0.0","0.0","0.0029114167854559173","0.010260657247901088","1.348489540060868","0.0","0.003833456119689838","0.0","0.0032382807585900437","0.008608254171218993","0.007928828044131218","0.0028493281809777366","0.0027873338673183225","0.003420832265462995","0.002285070372617045","0.008900150561499074","0.009003877508753227","0.013059446776572923","0.011103610207449996","0.007598401439603328","0.013596613875924366","0.011084121094273973","0.010607053111143525","0.009620235767829567","0.008259678006502619","0.010731665771685451","0.009363006763018657","0.0021852818767245097","0.009091748268572158","0.025714265576588485","0.004480554483528854","0.009061189535661485","0.008869958469754202","0.002361030920067052","0.0022601284491659272","0.0018391542634132708","0.0034476100080681558","0.0024458295985315373","0.004293173395970161","0.004579214352357212","0.0025923332971131054","0.004795045874173534","0.004151013688821708","0.0226276447864268","0.01462254664631181","0.01602693686543622","0.015326282883084562","0.014428141030373473","0.002019182721147269","0.0014883832205133188","0.0009832831099109687","0.0037659511330376417","0.0016345295470368193","0.003178818200828604","0.004094191659114589","0.0024912191401879813","0.0032647649405574786","0.003910253340981285","0.01543028339321429","0.01485537124391433","0.015675234074991284","0.01289703559343365","0.016369611794000778"]
        total_weight["fit_error"] = ferror
        total_weight["sum_wt"] = self.sum_wt
        if mode == "pass":
            onp.savez("output/draw/weight_kk_kk.npz", **total_weight)
        if mode == "truth":
            onp.savez("output/draw/weight_kk_kk.npz".replace(".","_truth."), **total_weight)
        print("total fit fraction:",total_fit_frac)
        print("run over , save data to weight.npz")
        return values, ferror, total_fit_frac




class Control(object):
    def __init__(self, args):
        self.bic_delete_mods = args.bic_delete_mods
        self.total_gpu_id = args.total_gpu_id
        self.processes_gpus = args.processes_gpus
        self.max_processes = args.max_processes
        self.max_processes_memory = args.max_processes_memory
        self.thread_gpus = args.thread_gpus
        self.threads_in_one_gpu = args.threads_in_one_gpu
        
        self.data_config = dict()
        self.data_config["data_slices"] = args.data_slices
        self.data_config["mc_slices"] = args.mc_slices
        self.data_config["mini_run"] = args.mini_run
        
        self.args_list = onp.array([1.02, 0.004, 0.9794812115574156, 0.10678616326827592, 8.570187550432664, 0.1, 0.1065182971468388, 0.1, 0.03807025376236671, 1.6761965590304995, 0.16270071108440043, 0.02201375198386279, 0.007433204877151768, 0.008302300118288414, -0.018544178045626723, 1.2896149318644679, 0.1959796900380022, -0.013533706721054747, -0.01650921272412254, 0.015339403283337268, 0.029798824827026338, 0.020982478502465995, 0.05458389002821978, 0.016032334798079934, 0.017179622009723918, -0.050143087437086675, -0.008718872479379924, 1.5222602842746435, 2.1619576785269476, 2.547889297662712, 0.08576525399315078, 0.15906049251159413, 0.324001266488012, -0.005345214125595505, 0.0031770703345810553, -0.0036743677603351056, 0.004813366936315499, 0.010000673114238258, 0.004643720949518616, -0.0009682564746806725, -0.0029804674908844213, 0.008315390493289363, 0.0005819695034846763, -0.15266341362240637, -0.05030214288962155, -0.01856511044577769, 0.0908719088071242, 0.029544599934778714, -0.010529063356530807, 0.003910028530572422, 0.0031787953173277725, 0.0333658928584713, -0.006053516058970728, 0.00545992748088964, -0.012498776031677225, 0.002196563552197887, -0.016814307435916394, -0.007232946300343621, 0.06277085590848677, -0.0031366601030189344, 0.13756083806604205, -0.0033564526519642953, 0.06849361819185686])
        self.float_list = onp.array([2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62])

        max_processes = self.max_processes
        self.process_pool = [Process()] * max_processes
        self.kk_data_lh = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])
        self.kk_mc_lh = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])

        self.kk_data_grad = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_grad = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_grad_v = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])

        self.kk_data_hvp = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_hvp = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_hvp_g = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_hvp_v = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])
        

    def grad_double(self,a,b,x):
        return (b-a)/2.0*onp.cos(x)
    
    def grad_a(self,x):
        return x/onp.sqrt(onp.power(x,2)+1)

    def grad_b(self,x):
        return -1.0*x/onp.sqrt(onp.power(x,2)+1)

    def likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        
            self.kk_data_lh[device_rank,_] = self.pwaf_list[device_rank][_].jit_data_likelihood_kk(args_float)   
            self.kk_mc_lh[device_rank,_] = self.pwaf_list[device_rank][_].jit_mc_likelihood_kk(args_float)   
        

    def thread_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
            threading_list[_].start()
        for _ in range(self.thread_gpus):
            threading_list[_].join()
        
        # result_kk = onp.sum(self.kk_data_lh) + 100.0*onp.log(onp.sum(self.kk_mc_lh))
        result_kk = onp.sum(self.kk_data_lh) + 100.0*onp.log(onp.sum(self.kk_mc_lh)/1000.0)
        
        result =+ result_kk
        return result

    def jit_grad_likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        
            self.kk_data_grad[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_grad_data_likelihood_kk(args_float)
            self.kk_mc_grad_v[device_rank,_], self.kk_mc_grad[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_grad_mc_likelihood_kk(args_float)
        

    def thread_grad_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        result = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.jit_grad_likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
            threading_list[_].start()
        for _ in range(self.thread_gpus):
            threading_list[_].join()
        
        result_kk = onp.sum(self.kk_data_grad,axis=(0,1)) + 100.0*onp.sum(self.kk_mc_grad,axis=(0,1))/onp.sum(self.kk_mc_grad_v)
        
        result =+ result_kk
        return result

    def mc_likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        
            self.kk_mc_hvp_v[device_rank,_] = self.pwaf_list[device_rank][_].jit_mc_likelihood_kk(args_float)
        

    def thread_mc_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.mc_likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
            threading_list[_].start()
        for _ in range(self.thread_gpus):
            threading_list[_].join()
        
        self.kk_mc_hvp_value = onp.sum(self.kk_mc_hvp_v)
        

    def jit_hvp_in_sigle_device(self, args_float, any_vector, device_rank):
        for _ in range(self.threads_in_one_gpu):
        
            self.kk_data_hvp[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_hvp_data_fwdrev_kk(args_float, any_vector)[1]
            self.kk_mc_hvp_g[device_rank,_,:], self.kk_mc_hvp[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_hvp_mc_fwdrev_kk(args_float, any_vector)
        

    def thread_hvp(self, args_float, any_vector):
        threading_list = [ None for _ in range(self.thread_gpus)]
        result = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.jit_hvp_in_sigle_device, args = (args_float, any_vector, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
            threading_list[_].start()
        for _ in range(self.thread_gpus):
            threading_list[_].join()
        self.thread_mc_likelihood(args_float)
        
        result_kk = onp.sum(self.kk_data_hvp, axis=(0,1)) + 100.0*(onp.sum(self.kk_mc_hvp,axis=(0,1))/self.kk_mc_hvp_value - onp.einsum("ikm,jln,n->m",self.kk_mc_hvp_g,self.kk_mc_hvp_g,any_vector)/(self.kk_mc_hvp_value**2))
        
        result =+ result_kk
        return result
    
    def require_compile(self, fcn, *args):
        logger.info("{} compile start".format(fcn.__name__))
        t1 = time.time()
        fcn(*args)
        t2 = time.time()
        logger.info("{} compile complete, time is {}".format(fcn.__name__,t2-t1))

    def my_callback(self, xk):
        logger.info(" fcn: {}".format(self.thread_likelihood(xk)))

    def compile_func(self):
        start = time.time()
        vector = onp.ones(self.args_float.shape)
        compile_likelihood = Thread(target = self.require_compile, args = (self.thread_likelihood, self.args_float))
        compile_grad = Thread(target = self.require_compile, args = (self.thread_grad_likelihood, self.args_float))
        compile_hvp = Thread(target = self.require_compile, args = (self.thread_hvp, self.args_float, vector))

        compile_hvp.start()
        compile_likelihood.start()
        compile_grad.start()

        compile_hvp.join()
        compile_likelihood.join()
        compile_grad.join()
        stop = time.time()
        logger.info("compile time: {}".format(stop-start))

    def run(self, process_initializer, process_returns):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x:str(x),process_initializer[0][0].total_gpu_id))
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=str(process_initializer[0][0].gpu_memory_limit_percentage)
        # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # 用多少拿多少的选项
        config.update("jax_enable_x64", True)

        logger.info("fit_pull_sample begins!!!!")

        start_time = time.time()

        device_list = jdevices()
        self.pwaf_list = [[PWAFunc(process_initializer[i][j], device_list[process_initializer[i][j].gpu_id]) for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
        for i in range(self.thread_gpus):
            for j in range(self.threads_in_one_gpu):
                self.pwaf_list[i][j].jit_request()
        
        pwaf = self.pwaf_list[0][0]
        fvalue, ferror, sof = pwaf.run_weight(self.args_list, self.float_list,mode="pass")
        fvalue, ferror, sof = pwaf.run_weight(self.args_list, self.float_list,mode="truth")
        self.save_in_json(fvalue,ferror,"output/error",str(0))
        sof = onp.array(sof)
        fvalue = onp.array(fvalue)
        min_fcn = [sof]

        stop_time = time.time()
        logger.info("=================set return=====================")
        process_returns.set("parameter_values", fvalue)
        process_returns.set("parameter_errors", ferror)
        process_returns.set("min_fcn", min_fcn)
        process_returns.info()
        process_returns.set("timer", stop_time - start_time)
        process_returns.set("process_id", os.getpid())
        process_returns.set("gpu_id", process_initializer[0][0].total_gpu_id)

        return

    def run_multiprocess(self):
        self.process_initializer = Process_Initializer_Generator(**self.data_config)
        process_initializer_generator = self.process_initializer.process_initializer_generator()
        self.process_returns = []
        self.start_time = time.time()
        while True:
            try:
                for _proc_numbering, _proc in enumerate(self.process_pool):
                    if not _proc.is_alive():
                        try:
                            _proc.join()
                        except AssertionError:
                            pass
                        finally:
                            # 创建多线程拆分模型, 注意! 一个进程只可以用来计算一个完整的拟合
                            # 线程模型是[i, j]形状, 对应[gpus, thread_in_gpus], 总线程为 gpus*thread_in_gpus
                            _proc_initializer_list = [[None for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
                            for i in range(self.thread_gpus):
                                for j in range(self.threads_in_one_gpu):
                                    with next(process_initializer_generator) as _proc_initializer_list[i][j]:
                                        logger.info("data: {}".format(_proc_initializer_list[i][j].data_numbering))
                                        _proc_initializer_list[i][j].gpu_memory_limit_percentage = self.max_processes_memory 
                                        _proc_initializer_list[i][j].gpu_id = i + _proc_numbering % self.processes_gpus # gpu的分配靠 thread_gpus 和 processes_gpus 和进程数调节
                                        _proc_initializer_list[i][j].total_gpu_id = self.total_gpu_id
                            
                            _proc_returns = ProcessReturns()
                            _proc_returns.set("data_numbering", [[_proc_initializer_list[i][j].data_numbering for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)])
                            self.process_pool[_proc_numbering] = Process(target=self.run, args=(_proc_initializer_list, _proc_returns))
                            self.process_pool[_proc_numbering].start()
                            self.process_returns.append(_proc_returns)
                            logger.info("=========== chambering data_numbering.{} ============".format(_proc_returns.get("data_numbering")))

            except StopIteration:
                for _ in self.process_pool:
                    try:
                        _.join()
                    except AssertionError:
                        pass
                break

        self.stop_time = time.time()
        logger.info("Total time: {}".format(self.stop_time - self.start_time))
    
    def get_result_dict(self):
        self.fvalues = (self.process_returns[0]._dict)["parameter_values"]
        self.fcn = (self.process_returns[0]._dict)["min_fcn"]

    def save_in_npz(self):
        logger.info("save in npz")
        total_values = []
        total_errors = []
        for _ in self.process_returns:
            total_values.append(_._dict["parameter_values"])
            total_errors.append(_._dict["parameter_errors"])
        np_values = onp.squeeze(onp.array(total_values), axis=None)
        np_errors = onp.squeeze(onp.array(total_errors), axis=None)
        logger.debug("{}".format(np_values.shape))
        logger.debug("{}".format(np_errors.shape))
        if total_values[0] is not None or total_errors[0] is not None:
            onp.savez("['output/draw/weight_kk_kk.npz']", fvalue=np_values, ferror=np_errors)

        logger.info("program over")
    
    def save_in_json(self, np_values, np_errors, save_addr = "", n = "",result_info={}):
        logger.info("save in json")
        json_addr_list = []
        
        json_addr_list.append("pwa_info_kk.json")
        
        name_list = ['phi_m', 'phi_w', 'kk_f980_m', 'kk_g_kk', 'kk_rg', 'kk_f980_c1', 'kk_f980_c2', 'kk_f980_t1', 'kk_f980_t2', 'kk_f1710_m', 'kk_f1710_w', 'kk_f1710_c1', 'kk_f1710_c2', 'kk_f1710_t1', 'kk_f1710_t2', 'kk_f1270_m', 'kk_f1270_w', 'kk_f1270_c1', 'kk_f1270_c2', 'kk_f1270_c3', 'kk_f1270_c4', 'kk_f1270_c5', 'kk_f1270_t1', 'kk_f1270_t2', 'kk_f1270_t3', 'kk_f1270_t4', 'kk_f1270_t5', 'kk_f1525_m', 'kk_f2150_m', 'kk_f2340_m', 'kk_f1525_w', 'kk_f2150_w', 'kk_f2340_w', 'kk_f1525_c1', 'kk_f1525_c2', 'kk_f1525_c3', 'kk_f1525_c4', 'kk_f1525_c5', 'kk_f2150_c1', 'kk_f2150_c2', 'kk_f2150_c3', 'kk_f2150_c4', 'kk_f2150_c5', 'kk_f2340_c1', 'kk_f2340_c2', 'kk_f2340_c3', 'kk_f2340_c4', 'kk_f2340_c5', 'kk_f1525_t1', 'kk_f1525_t2', 'kk_f1525_t3', 'kk_f1525_t4', 'kk_f1525_t5', 'kk_f2150_t1', 'kk_f2150_t2', 'kk_f2150_t3', 'kk_f2150_t4', 'kk_f2150_t5', 'kk_f2340_t1', 'kk_f2340_t2', 'kk_f2340_t3', 'kk_f2340_t4', 'kk_f2340_t5']
        mod_name_list = ['phif0_980', 'phif0_1710', 'phif2_1270', 'phif2_1525', 'phif2_2150', 'phif2_2340']
        for json_addr in json_addr_list:
            filename = glob.glob(os.path.join("config/",json_addr+"*"))
            with open(filename[0], encoding='utf-8') as f:
                pwa_json = json.loads(f.read())
                mod_info = pwa_json["mod_info"]
                mod_index = [n for n, mod in enumerate(mod_info) if re.match(".*"+"_".join(mod["mod"].split("_")[0:-1])+".*"," ".join(mod_name_list))]
                mod_info = [mod_info[i] for i in mod_index]
                for i in range(len(name_list)):
                    for mod in mod_info:
                        for arg_key in mod["args"].keys():
                            if name_list[i] == arg_key:
                                mod["args"][arg_key]["value"] = float(np_values[i])
                                mod["args"][arg_key]["error"] = float(np_errors[i])
                _pwa_json = {"mod_info":mod_info} 
                json_str = json.dumps(_pwa_json, indent=4)
                with open(save_addr + "/{0}.{1}".format(json_addr, n), 'w') as json_file:
                    json_file.write(json_str)
        with open(save_addr + "/result_info.json", 'w') as f:
            json.dump(result_info,f)
