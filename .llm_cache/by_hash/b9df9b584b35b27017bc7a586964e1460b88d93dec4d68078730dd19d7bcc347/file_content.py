import os
import sys
import logging
import json
import logging.config
import numpy as onp
from functools import reduce

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)


from rendered_scripts import draw_lh_object_kk as draw_object_0


class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_draw.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)

logger = Logger("draw")
logger = logging.getLogger("draw")



class run_bic(object):
    def __init__(self):
        self.args = draw_object_0.args()
        self.args.total_gpu_id = [0, 1]
        self.args.processes_gpus = 2
        self.args.max_processes = 1
        self.args.max_processes_memory = 0.89
        self.args.thread_gpus = 1
        self.args.threads_in_one_gpu = 1
        
        self.args.data_slices = 1
        self.args.mc_slices = 1
        self.args.mini_run = 1
        
        self.BIC = 0.0
        self.AIC = 0.0
        self.fcn = 0.0
        self.all_mod_dict = dict()

    def run_obj(self,draw):
        draw.run_multiprocess()
        draw.get_result_dict()
        logger.debug("{}".format(draw.frac_dict))
        logger.debug("{}".format(draw.fcn))
        new_dict = dict(filter(lambda item: item[1][2] > 0.01, draw.frac_dict.items()))
        logger.debug("{}".format(new_dict))
        bic = 2.0*draw.fcn["fcn"] + onp.log(draw.fcn["data_size"])*onp.array(reduce(lambda x,y:x+y[0], new_dict.values(), 0))
        aic = 2.0*draw.fcn["fcn"] + 2.0*onp.array(reduce(lambda x,y:x+y[0], new_dict.values(), 0))
        # print(2.0*draw.fcn["fcn"])
        # print(onp.log(draw.fcn["data_size"]))
        # print(onp.array(reduce(lambda x,y:x+y[0], new_dict.values(), 0)))
        return bic, aic

    def run_bic(self):
        
        draw = draw_object_0.Control(self.args)
        bic, aic = self.run_obj(draw)
        logger.info("one part of BIC : {}".format(bic))
        self.BIC += bic
        self.AIC += aic
        self.all_mod_dict = {**self.all_mod_dict, **draw.frac_dict}
        self.fcn += draw.fcn["fcn"]
        
        self.all_mod_name = list(self.all_mod_dict.keys())
        logger.info("BIC : {}".format(self.BIC))
        logger.info("AIC : {}".format(self.AIC))

    def run_select_mods(self):
        name_temp = list()
        new_BIC = self.BIC/2.0
        delete_name = ""
        while True:
            for name in self.all_mod_name:
                logger.info("cycle delete mod : {}".format(name))
                temp_BIC = 0.0
                self.args.bic_delete_mods = list()
                for name in name_temp:
                    self.args.bic_delete_mods.append(name)
                self.args.bic_delete_mods.append(name)
                
                draw = draw_object_0.Control(self.args)
                bic = self.run_obj(draw)
                logger.info("one part of BIC : {}".format(bic))
                temp_BIC += bic
                
                logger.info("new BIC : {}".format(temp_BIC))
                if new_BIC > temp_BIC:
                    new_BIC = temp_BIC     
                    delete_name = name
                logger.info("mini BIC : {}".format(new_BIC))
                logger.info("delete mod : {}".format(delete_name))

            if new_BIC > self.BIC:
                name_temp.append(delete_name)
                break
            name_temp.append(delete_name)
            self.BIC = new_BIC
        logger.info("delete mods : {}".format(name_temp))
    
if __name__ == "__main__":
    # A = run_bic()
    # A.run_bic()
    # A.run_select_mods()
    args = draw_object_0.args()
    args.total_gpu_id = [0, 1]
    args.processes_gpus = 2
    args.max_processes = 1
    args.max_processes_memory = 0.89
    args.thread_gpus = 1
    args.threads_in_one_gpu = 1
    
    args.data_slices = 1
    args.mc_slices = 1
    args.mini_run = 1
    
    likelihood = 0.0
    
    draw = draw_object_0.Control(args)
    draw.run_multiprocess()
    draw.get_result_dict()
    likelihood += draw.fcn
    
    print("likelihood :",likelihood)