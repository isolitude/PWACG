import os, sys, json, logging, logging.config

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

class Logger():
    def __init__(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_select.json", "r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)

logger = Logger("select")

from rendered_scripts import select_object_kk as select_object_0

args = select_object_0.args()
args.max_processes = 1
args.max_processes_memory = 0.89
args.processes_gpus = 2
args.thread_gpus = 1
args.threads_in_one_gpu = 1
args.total_gpu_id = [0, 1]
args.data_slices = 1
args.mc_slices = 1
args.mini_run = 1

cl = select_object_0.Control(args)
cl.run_multiprocess()
os.system("cp output/fit/fit_result_kk/pwa_info* data/select/")
