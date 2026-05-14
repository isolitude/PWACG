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
from rendered_scripts import fit_object_kk as fit_object


class Logger:
    def __init__(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_batch.json", "r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)


logger = Logger("batch")


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
        self.data_phi_kk = None
        self.mc_phi_kk = None
        self.data_f_kk = None
        self.mc_f_kk = None
        self.data_phif0_kk = None
        self.mc_phif0_kk = None
        self.data_phif2_kk = None
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


class Process_Initializer_Generator:
    def __init__(self, data_slices=None, mc_slices=None, mini_run=None):
        self.data_slices = data_slices
        self.mc_slices = mc_slices
        self.mini_run = mini_run

    def data_npz(self, num):
        self.data_phi_kk = onp.load("data/real_data/phi_kk.npy")
        self.all_data_phi_kk = onp.array_split(self.data_phi_kk, num, axis=1)
        self.data_f_kk = onp.load("data/real_data/f_kk.npy")
        self.all_data_f_kk = onp.array_split(self.data_f_kk, num, axis=0)
        self.data_phif0_kk = onp.load("data/real_data/phif0_kk.npy")
        self.all_data_phif0_kk = onp.array_split(self.data_phif0_kk, num, axis=1)
        self.data_phif2_kk = onp.load("data/real_data/phif2_kk.npy")
        self.all_data_phif2_kk = onp.array_split(self.data_phif2_kk, num, axis=1)
        self.wt_data_kk = onp.load("data/weight/weight_kk.npy")
        self.all_wt_data_kk = onp.array_split(self.wt_data_kk, num, axis=0)

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
        regular_phi_kk = 1. / onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phi_kk) ** 2, axis=2)), axis=1)
        self.all_data_phi_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_data_phi_kk), regular_phi_kk)
        self.all_mc_phi_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_mc_phi_kk), regular_phi_kk)

        self.re_phif0_kk = onp.load("data/mc_truth/phif0_kk.npy")
        regular_phif0_kk = 1. / onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif0_kk) ** 2, axis=2)), axis=1)
        self.all_data_phif0_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_data_phif0_kk), regular_phif0_kk)
        self.all_mc_phif0_kk = onp.einsum("ijkl,j->ijkl", onp.array(self.all_mc_phif0_kk), regular_phif0_kk)

        self.re_phif2_kk = onp.load("data/mc_truth/phif2_kk.npy")
        regular_phif2_kk = 1. / onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif2_kk) ** 2, axis=2)), axis=1)
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

        # mw_index and mw_range for parameter scanning
        self.mw_index = onp.array([0, 5, 6, 11, 12, 23, 26, 24, 27, 25, 28])
        self.mw_range = onp.array([
            [0.98, 10.0],
            [1.704, 1.0],
            [0.123, 1.0],
            [1.2755, 1.0],
            [0.1867, 1.0],
            [1.517, 1.0],
            [0.086, 1.0],
            [2.157, 1.0],
            [0.152, 1.0],
            [2.345, 0.01],
            [0.322, 1.0],
        ])
        self.scan_steps = 5  # number of steps per dimension (can be customized)

        # Grid parameters (only scan over first few mass/width for demonstration)
        self.scan_indices = [0, 5, 6, 11, 12]  # phi mass, and kk_f0 mass/width, kk_f1270 mass/width
        self.grid_points = self._generate_grid()

        # Storage for results
        self.results = []

    def _generate_grid(self):
        """Generate a grid of parameter values for scanning."""
        # For each scan index, create linear grid from center - range to center + range
        centers = onp.array([1.02, 1.704, 0.123, 1.2755, 0.1867])
        ranges = onp.array([0.02, 0.1, 0.02, 0.1, 0.02])  # small ranges for demonstration
        steps = self.scan_steps
        axes = [onp.linspace(centers[i] - ranges[i], centers[i] + ranges[i], steps) for i in range(len(self.scan_indices))]
        mesh = onp.meshgrid(*axes, indexing='ij')
        points = onp.stack(mesh, axis=-1).reshape(-1, len(self.scan_indices))
        return points

    def run(self, initializer, returns):
        """Run a fit for a single grid point."""
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x: str(x), initializer[0][0].total_gpu_id))
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(initializer[0][0].gpu_memory_limit_percentage)
        config.update("jax_enable_x64", True)
        device_list = jdevices()

        # Extract base parameters from initializer (if needed) – here we use default fit initial params
        # For demonstration, we create fit Control and modify initial parameters
        fit_args = fit_object.args()
        # Set all run_config and data_config from this batch args
        fit_args.max_processes = self.max_processes
        fit_args.max_processes_memory = self.max_processes_memory
        fit_args.processes_gpus = self.processes_gpus
        fit_args.thread_gpus = self.thread_gpus
        fit_args.threads_in_one_gpu = self.threads_in_one_gpu
        fit_args.total_gpu_id = self.total_gpu_id
        fit_args.data_slices = self.data_slices
        fit_args.mc_slices = self.mc_slices
        fit_args.mini_run = self.mini_run

        # Modify initial parameters for this grid point
        base_params = onp.array(initializer[0][0].base_params) if hasattr(initializer[0][0], 'base_params') else None
        if base_params is None:
            base_params = onp.array([
                1.02, 0.004, 0.9794812115574156, 0.10678616326827592, 8.570187550432664,
                0.1, 0.1065182971468388, 0.1, 0.03807025376236671, 1.6761965590304995,
                0.16270071108440043, 0.02201375198386279, 0.007433204877151768, 0.008302300118288414,
                -0.018544178045626723, 1.2896149318644679, 0.1959796900380022, -0.013533706721054747,
                -0.01650921272412254, 0.015339403283337268, 0.029798824827026338, 0.020982478502465995,
                0.05458389002821978, 0.016032334798079934, 0.017179622009723918, -0.050143087437086675,
                -0.008718872479379924, 1.5222602842746435, 2.1619576785269476, 2.547889297662712,
                0.08576525399315078, 0.15906049251159413, 0.324001266488012, -0.005345214125595505,
                0.0031770703345810553, -0.0036743677603351056, 0.004813366936315499, 0.010000673114238258,
                0.004643720949518616, -0.0009682564746806725, -0.0029804674908844213, 0.008315390493289363,
                0.0005819695034846763, -0.15266341362240637, -0.05030214288962155, -0.01856511044577769,
                0.0908719088071242, 0.029544599934778714, -0.010529063356530807, 0.003910028530572422,
                0.0031787953173277725, 0.0333658928584713, -0.006053516058970728, 0.00545992748088964,
                -0.012498776031677225, 0.002196563552197887, -0.016814307435916394, -0.007232946300343621,
                0.06277085590848677, -0.0031366601030189344, 0.13756083806604205, -0.0033564526519642953,
                0.06849361819185686
            ])
        # Overwrite the scanned parameters
        for idx, param_val in zip(self.scan_indices, self.grid_point):
            base_params[idx] = param_val
        fit_args.args_list = base_params
        fit_args.float_list = onp.array([2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                          23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                          40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                                          57, 58, 59, 60, 61, 62])
        fit_control = fit_object.Control(fit_args)
        fit_control.run(initializer, returns)

    def run_multiprocess(self):
        """Main entry: generate grid points and run fits in parallel."""
        process_generator = Process_Initializer_Generator(
            data_slices=self.data_slices,
            mc_slices=self.mc_slices,
            mini_run=self.mini_run
        )
        # Prepare initial processers for each GPU device
        mgr = Manager()
        process_initializers = []
        for gpuid in self.total_gpu_id:
            for _ in range(self.thread_gpus * self.threads_in_one_gpu):
                init = ProcessInitializers()
                init.total_gpu_id = self.total_gpu_id
                init.gpu_memory_limit_percentage = self.max_processes_memory
                init.gpu_id = gpuid
                # Attach base parameters (can be modified later)
                init.base_params = onp.array([
                    1.02, 0.004, 0.9794812115574156, 0.10678616326827592, 8.570187550432664,
                    0.1, 0.1065182971468388, 0.1, 0.03807025376236671, 1.6761965590304995,
                    0.16270071108440043, 0.02201375198386279, 0.007433204877151768, 0.008302300118288414,
                    -0.018544178045626723, 1.2896149318644679, 0.1959796900380022, -0.013533706721054747,
                    -0.01650921272412254, 0.015339403283337268, 0.029798824827026338, 0.020982478502465995,
                    0.05458389002821978, 0.016032334798079934, 0.017179622009723918, -0.050143087437086675,
                    -0.008718872479379924, 1.5222602842746435, 2.1619576785269476, 2.547889297662712,
                    0.08576525399315078, 0.15906049251159413, 0.324001266488012, -0.005345214125595505,
                    0.0031770703345810553, -0.0036743677603351056, 0.004813366936315499, 0.010000673114238258,
                    0.004643720949518616, -0.0009682564746806725, -0.0029804674908844213, 0.008315390493289363,
                    0.0005819695034846763, -0.15266341362240637, -0.05030214288962155, -0.01856511044577769,
                    0.0908719088071242, 0.029544599934778714, -0.010529063356530807, 0.003910028530572422,
                    0.0031787953173277725, 0.0333658928584713, -0.006053516058970728, 0.00545992748088964,
                    -0.012498776031677225, 0.002196563552197887, -0.016814307435916394, -0.007232946300343621,
                    0.06277085590848677, -0.0031366601030189344, 0.13756083806604205, -0.0033564526519642953,
                    0.06849361819185686
                ])
                # Load data for this initializer (placeholder - actual loading happens later)
                process_initializers.append([init])

        # Run fits for each grid point
        self.process_pool = []
        for point_idx, grid_point in enumerate(self.grid_points):
            self.grid_point = grid_point
            returns = ProcessReturns()
            p = Process(target=self.run, args=(process_initializers, returns))
            p.start()
            self.process_pool.append((p, returns))
            if (point_idx + 1) % self.max_processes == 0:
                for p, ret in self.process_pool:
                    p.join()
                    self.results.append({
                        'grid_point': ret.get('grid_point', [0]),
                        'min_fcn': ret.get('min_fcn', None),
                        'parameter_values': ret.get('parameter_values', None),
                        'parameter_errors': ret.get('parameter_errors', None)
                    })
                self.process_pool = []

        # Remaining processes
        for p, ret in self.process_pool:
            p.join()
            self.results.append({
                'grid_point': ret.get('grid_point', [0]),
                'min_fcn': ret.get('min_fcn', None),
                'parameter_values': ret.get('parameter_values', None),
                'parameter_errors': ret.get('parameter_errors', None)
            })

        # Save results to JSON
        result_path = "output/batch/batch_result_kk.json"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Batch scan complete. Results saved to {}".format(result_path))


if __name__ == '__main__':
    # Command-line execution (optional)
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