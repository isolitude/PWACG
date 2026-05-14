import os, sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)
from rendered_scripts import fit_object_kk as fit_object
if __name__ == '__main__':
    logger = fit_object.Logger("fit")
    args = fit_object.args()
    args.max_processes = 1
    args.max_processes_memory = 0.89
    args.processes_gpus = 2
    args.thread_gpus = 1
    args.threads_in_one_gpu = 1
    args.total_gpu_id = [0, 1, 2, 3]
    args.data_slices = 1
    args.mc_slices = 1
    args.mini_run = 1
    cl = fit_object.Control(args)
    cl.run_multiprocess()