import os
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from rendered_scripts import pull_object_kk as pull_object

if __name__ == '__main__':
    args = pull_object.args()
    args.total_gpu_id = [0, 1]
    args.processes_gpus = 2
    args.max_processes = 1
    args.max_processes_memory = 0.86
    args.thread_gpus = 1
    args.threads_in_one_gpu = 1
    
    args.data_slices = 1
    args.mc_slices = 1
    args.mini_run = 1
    
    
    cl = pull_object.Control(args)
    cl.run_multiprocess()
    
    cl.save_in_npz()
    