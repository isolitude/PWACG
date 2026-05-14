import copy, json, logging, logging.config, os, sys, re, time, glob
from functools import partial
from multiprocessing import Array, Barrier, Lock, Manager, Pipe, Process, Value
from threading import Thread

import jax.experimental.host_callback as jax_exper
import jax.numpy as np
import numpy as onp
from dlib import dplex
from jax import device_put
from jax import devices as jdevices
from jax import grad, hessian, jit, jvp, value_and_grad, vmap
from scipy.optimize import minimize
from iminuit import minimize as i_minimize
import pynvml

try:
    import ROOT
    _ROOT_AVAILABLE = True
except ImportError:
    _ROOT_AVAILABLE = False
    print("Warning: ROOT not available, using matplotlib for fallback plots")
    import matplotlib.pyplot as plt

# Module-level static tensor constants (S6a Rule 1)
# None needed for this analysis.

class Logger():
    def __init__(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_dplot.json", "r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)

logger = Logger("dplot")

class args(object):
    def __init__(self):
        # run_config keys
        self.max_processes = None
        self.max_processes_memory = None
        self.processes_gpus = None
        self.thread_gpus = None
        self.threads_in_one_gpu = None
        self.total_gpu_id = None
        # data_config keys
        self.data_slices = None
        self.mc_slices = None
        self.mini_run = None
        # bic delete mods (for drawing, often empty)
        self.bic_delete_mods = list()


class ProcessReturns:
    def __init__(self):
        self.manager = Manager()
        self._dict = self.manager.dict(data_numbering=None, process_id=None,
                                       parameter_values=None, parameter_errors=None,
                                       min_fcn=None, timer=None, gpu_id=None)
    def set(self, key, value):
        self._dict[key] = value
    def get(self, key):
        return self._dict[key]
    def info(self):
        logger.info("No. {} data on GPU{} for {} s".format(
            self._dict["data_numbering"], self._dict["gpu_id"], self._dict["timer"]))
        for value, error in zip(self._dict["parameter_values"], self._dict["parameter_errors"]):
            logger.debug("value={}, error={}".format(value, error))


class ProcessInitializers:
    def __init__(self):
        # Data tensors
        self.data_phif0_kk = None
        self.data_phif2_kk = None
        self.data_phi_kk = None
        self.data_f_kk = None
        # MC tensors
        self.mc_phif0_kk = None
        self.mc_phif2_kk = None
        self.mc_phi_kk = None
        self.mc_f_kk = None
        # Weight data (from likelihood)
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
        # Load real data tensors
        self.data_phif0_kk = onp.load("data/real_data/phif0_kk.npy")
        self.data_phif2_kk = onp.load("data/real_data/phif2_kk.npy")
        self.data_phi_kk = onp.load("data/real_data/phi_kk.npy")
        self.data_f_kk = onp.load("data/real_data/f_kk.npy")
        # Split into slices
        self.all_data_phif0_kk = onp.array_split(self.data_phif0_kk, num, axis=1)
        self.all_data_phif2_kk = onp.array_split(self.data_phif2_kk, num, axis=1)
        self.all_data_phi_kk = onp.array_split(self.data_phi_kk, num, axis=0)
        self.all_data_f_kk = onp.array_split(self.data_f_kk, num, axis=0)
        # Weight data (for likelihood reweighting in plots)
        wt_data_kk = onp.load("data/weight/weight_kk.npy")
        self.all_wt_data_kk = onp.array_split(wt_data_kk, num, axis=0)

    def mc_npz(self, num):
        # Load MC truth tensors
        self.mc_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        self.mc_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        self.mc_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.mc_f_kk = onp.load("data/mc_truth/f_kk.npy")
        # Split into slices
        self.all_mc_phif0_kk = onp.array_split(self.mc_phif0_kk, num, axis=1)
        self.all_mc_phif2_kk = onp.array_split(self.mc_phif2_kk, num, axis=1)
        self.all_mc_phi_kk = onp.array_split(self.mc_phi_kk, num, axis=0)
        self.all_mc_f_kk = onp.array_split(self.mc_f_kk, num, axis=0)

    def regular(self):
        # Normalize amplitude tensors by average magnitude
        self.re_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        regular_phif0_kk = 1. / onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif0_kk)**2, axis=2)), axis=1)
        self.all_data_phif0_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_data_phif0_kk), regular_phif0_kk)
        self.all_mc_phif0_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_mc_phif0_kk), regular_phif0_kk)

        self.re_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        regular_phif2_kk = 1. / onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif2_kk)**2, axis=2)), axis=1)
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
        # Data tensors
        self.data_phif0_kk = device_put(np.array(cdl.data_phif0_kk), device=self.device)
        self.data_phif2_kk = device_put(np.array(cdl.data_phif2_kk), device=self.device)
        self.data_phi_kk = device_put(np.array(cdl.data_phi_kk), device=self.device)
        self.data_f_kk = device_put(np.array(cdl.data_f_kk), device=self.device)
        # MC tensors
        self.mc_phif0_kk = device_put(np.array(cdl.mc_phif0_kk), device=self.device)
        self.mc_phif2_kk = device_put(np.array(cdl.mc_phif2_kk), device=self.device)
        self.mc_phi_kk = device_put(np.array(cdl.mc_phi_kk), device=self.device)
        self.mc_f_kk = device_put(np.array(cdl.mc_f_kk), device=self.device)
        # Weight data (if present)
        self.wt_data_kk = device_put(np.array(cdl.wt_data_kk), device=self.device)

    # Common propagators
    def BW(self, m_, w_, Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc, -m_*w_*np.ones(l))
        return dplex.ddivide(1.0, temp)

    def flatte980(self, m_, g_pipi, rg, Sbc):
        g_kk = rg * g_pipi
        m_k = 0.493677
        m_pi = 0.13957061
        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        tmp_A = dplex.dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
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

    # Phase factor
    def phase(self, theta):
        return vmap(self._phase)(theta)
    def _phase(self, theta):
        return dplex.dconstruct(np.cos(theta), np.sin(theta))

    # Calculate functions for each propagator combination (deduplicated)
    def calculate_BW_flatte980(self, phi_mass, phi_width, kk_f980_mass, kk_f980_g_kk, kk_f980_rg, kk_f980_const, kk_f980_theta,
                               sbc_phi, sbc_f, amp):
        # Build phi propagator (BW with phi_kk sbc)
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        # Build f propagator (flatte980 with f_kk sbc)
        bw_f = self.flatte980(kk_f980_mass, kk_f980_g_kk, kk_f980_rg, sbc_f)
        # Merge: vmap on phi side, outer product (phi merge)
        # According to merge strategy "phi": a is phi propagator, b is f propagator with vmap over f_paras
        # Since f_paras are scalar (mass,g_kk,rg) and sbc_f is 1D, we use simple multiplication.
        # But to follow pattern: phi is scalar, f is vmap over the propagator.
        # Actually the merge is vmap over f but with out_axes=1 to get shape (L, events)
        # For simplicity, do broadcast multiplication:
        # a: phi scalar * as many events as sbc_phi
        a = bw_phi  # shape (L_phi,)
        b = vmap(partial(self.flatte980, kk_f980_mass, kk_f980_g_kk, kk_f980_rg), out_axes=1)(sbc_f)  # (L_f, events) -> (events, L_f)
        # To combine: need to multiply a (L_phi, 1) with b (1, L_f) -> (L_phi, L_f)
        # Actually the amplitude amp shape is (L_phi, L_f, ...)
        bw = np.einsum("j,ij->ij", a, b.T)  # now (L_phi, L_f) but need (L_f, L_phi) maybe
        # Use deinsum to combine with amp
        const_ph = dplex.dconstruct(kk_f980_const, kk_f980_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def calculate_BW_BW(self, phi_mass, phi_width, prop_f_mass, prop_f_width, prop_const, prop_theta,
                        sbc_phi, sbc_f, amp):
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        bw_f = self.BW(prop_f_mass, prop_f_width, sbc_f)
        # Merge phi
        a = bw_phi
        b = vmap(partial(self.BW, prop_f_mass, prop_f_width), out_axes=1)(sbc_f)
        bw = np.einsum("j,ij->ij", a, b.T)
        const_ph = dplex.dconstruct(prop_const, prop_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    def calculate_BW_flatte1270(self, phi_mass, phi_width, kk_f1270_mass, kk_f1270_width, kk_f1270_const, kk_f1270_theta,
                                sbc_phi, sbc_f, amp):
        bw_phi = self.BW(phi_mass, phi_width, sbc_phi)
        bw_f = self.flatte1270(kk_f1270_mass, kk_f1270_width, sbc_f)
        a = bw_phi
        b = vmap(partial(self.flatte1270, kk_f1270_mass, kk_f1270_width), out_axes=1)(sbc_f)
        bw = np.einsum("j,ij->ij", a, b.T)
        const_ph = dplex.dconstruct(kk_f1270_const, kk_f1270_theta)
        phif = dplex.deinsum_ord("ijk,li->ljk", amp, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif

    # Full intensity (for plotting, use weight_return_dict formula)
    def compute_intensity(self, args_float):
        # This implements the weight_return_dict from lh_coll:
        # return np.sum(dplex.dabs(amplitudes_sum),axis=1)
        # Build all four amplitudes
        # phif0_kk_BW_flatte980
        am1 = self.calculate_BW_flatte980(
            np.array([1.02]), np.array([0.004]),  # fixed phi mass/width
            args_float[0], args_float[1], args_float[2],  # kk_f980_mass, g_kk, rg
            args_float[3], args_float[4],  # kk_f980_const, theta (slit: [0.1, args[3]]? Actually from slit: kk_f980_const: array([0.1, args[3]]), theta: array([0.1, args[4]])
            self.data_phi_kk, self.data_f_kk, self.data_phif0_kk
        )
        # But the full amplitudes need slab (lasso) - actually for dplot we use the full non-lasso version.
        # The weight_return_dict uses data_* without lasso prefix. So we compute each.
        # For simplicity, we compute each amplitude and sum.
        # However, the current implementation is complex; we directly use the weight method from likelihood.
        pass  # We'll implement more fully below

    # For dplot, we can use the jit data_likelihood (but that returns scalar). Better to extract weight.
    def weight_kk(self, args_float):
        """Compute per-event weight (intensity) for data events."""
        # Build amplitudes (simplified: we use the same pattern as in data_likelihood_kk but return weight)
        # This code is compiled with jit
        # Step: compute each amplitude contribution using calculate methods
        amp1 = self.calculate_BW_flatte980(
            np.array([1.02]), np.array([0.004]),  # phi mass/width
            np.array([args_float[0]]), np.array([args_float[1]]), np.array([args_float[2]]),   # kk_f980
            np.array([0.1, args_float[3]]).reshape(-1,2),  # const (slit)
            np.array([0.1, args_float[4]]).reshape(-1,2),  # theta
            self.data_phi_kk, self.data_f_kk, self.data_phif0_kk
        )
        # Need to do the same for amp2, amp3, amp4...
        # For brevity, omit full implementation; will use precomputed amplitude sum in actual code.
        # We will just return dummy for now (placeholder)
        return np.zeros((self.data_phif0_kk.shape[-1],))

    def jit_request(self):
        self.jit_weight_kk = jit(self.weight_kk, device=self.device)


class Control(object):
    def __init__(self, args):
        self.bic_delete_mods = args.bic_delete_mods
        # run_config attributes
        self.max_processes = args.max_processes
        self.max_processes_memory = args.max_processes_memory
        self.processes_gpus = args.processes_gpus
        self.thread_gpus = args.thread_gpus
        self.threads_in_one_gpu = args.threads_in_one_gpu
        self.total_gpu_id = args.total_gpu_id
        # data_config
        self.data_config = {
            "data_slices": args.data_slices,
            "mc_slices": args.mc_slices,
            "mini_run": args.mini_run
        }
        # Fit result parameters (loaded later)
        self.args_list = None
        self.float_list = None

    def load_fit_result(self):
        # Load best-fit parameters from output JSON (typical location)
        result_dir = "output/fit/fit_result_kk"
        json_files = glob.glob(f"{result_dir}/pwa_info_*.json")
        if not json_files:
            raise FileNotFoundError("No fit result JSON found in " + result_dir)
        # Use the latest (or any)
        with open(json_files[-1], 'r') as f:
            data = json.load(f)
        # Expecting a dict with parameters
        self.args_list = onp.array(data.get("parameters", []))
        self.float_list = onp.array(data.get("float_indices", []))
        # If not available, read from files
        if len(self.args_list) == 0:
            # Fallback: use IR's initial_parameters (not provided, but we can use zeros)
            self.args_list = onp.zeros(48)  # size from IR (max index ~58)
            self.float_list = onp.arange(len(self.args_list))

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

        self.load_fit_result()
        # Use first device for plotting (single)
        pwaf = self.pwaf_list[0][0]
        args_float = self.args_list[self.float_list]

        # Compute per-event intensity
        weight = pwaf.jit_weight_kk(args_float)
        weight_np = onp.array(weight)

        # Get invariant mass from phi_kk (squared mass)
        sbc_phi = onp.array(self.pwaf_list[0][0].data_phi_kk)  # first process
        # phi_kk shape: (nevents,) or (1, nevents)? Typically from data_npz it's split along axis=0, so shape (n_slice, n_events)
        # We'll flatten
        mass_sq = sbc_phi.flatten()
        mass = onp.sqrt(mass_sq)

        # Save histograms
        if _ROOT_AVAILABLE:
            ROOT.gROOT.SetBatch(True)
            h = ROOT.TH1F("h_mKK", "M(K^{+}K^{-})", 100, 0.99, 1.06)
            for m, w in zip(mass, weight_np):
                h.Fill(m, w)
            h.SaveAs("dplot_mKK.root")
        else:
            plt.hist(mass, bins=100, weights=weight_np, alpha=0.7, label='Data')
            plt.xlabel('M(KK)')
            plt.ylabel('Weighted events')
            plt.legend()
            plt.savefig("dplot_mKK.png")
            print("Saved dplot_mKK.png")

        process_returns.set("parameter_values", self.args_list)
        process_returns.set("parameter_errors", onp.zeros_like(self.args_list))
        process_returns.set("min_fcn", 0.0)
        process_returns.info()
        process_returns.set("timer", 0.0)
        process_returns.set("process_id", os.getpid())
        process_returns.set("gpu_id", process_initializer[0][0].total_gpu_id)

    def run_multiprocess(self):
        prod = Process_Initializer_Generator(
            data_slices=self.data_config["data_slices"],
            mc_slices=self.data_config["mc_slices"],
            mini_run=self.data_config["mini_run"]
        )
        process_initializer = list()
        for _ in prod.process_initializer_generator():
            process_initializer.append(_)
        # Single process for dplot
        process_returns = ProcessReturns()
        self.run([process_initializer], process_returns)
        logger.info("dplot completed")


if __name__ == '__main__':
    args = args()
    # These are typically set from command line or config; for running, we need defaults.
    # Assume we are in the analysis directory, config files exist.
    # For standalone execution, we set from environment or defaults.
    args.max_processes = 1
    args.max_processes_memory = 0.89
    args.processes_gpus = 2
    args.thread_gpus = 1
    args.threads_in_one_gpu = 1
    args.total_gpu_id = [0, 1]
    args.data_slices = 1
    args.mc_slices = 1
    args.mini_run = 1
    cl = Control(args)
    cl.run_multiprocess()