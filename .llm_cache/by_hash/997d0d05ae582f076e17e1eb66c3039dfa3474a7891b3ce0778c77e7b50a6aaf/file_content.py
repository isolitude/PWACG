import os
import sys
import time
import logging
import logging.config
import json
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)
os.system('rm -rf output/pictures/*/*')

class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_draw.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)

logger = Logger("draw")


from rendered_scripts import dplot_object_kk as dplot_object

####################################################################################







## 拟合结果
mods = dplot_object.Draw_Mods()
mods.draw_mods()
## 预设值
# mods = dplot_object.Draw_Mods(args=args, defult=True)
# mods.draw_mods()

####################################################################################


# dplot_object.Draw_correlation()

localtime = time.localtime(time.time())
new_dir = 'result_repo/{:0>2d}{:0>2d}-{:0>2d}{:0>2d}'.format(localtime[1],localtime[2],localtime[3],localtime[4])
os.system('rm -rf {}'.format(new_dir))
os.system('mkdir -p {}/output'.format(new_dir))
os.system('cp -r output/draw/*.md {}/output/'.format(new_dir))
os.system('cp -r output/draw/*.latex {}/output/'.format(new_dir))
os.system('cp -r output/fit/* {}/output'.format(new_dir))
os.system('cp -r output/pictures/ {}'.format(new_dir))
os.system('cp -r config/ {}'.format(new_dir))