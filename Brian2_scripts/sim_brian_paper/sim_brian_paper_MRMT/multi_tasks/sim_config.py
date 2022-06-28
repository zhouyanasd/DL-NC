import os, sys
import numpy as np
from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks import *

# --- tasks settings ---
tasks = {0: {'name':'HAPT', 'evaluator': task_HAPT_evaluator},
         1: {'name':'KTH', 'evaluator': task_KTH_evaluator},
         2: {'name':'NMNIST', 'evaluator': task_NMNIST_evaluator}}

# -------path settings------------
exec_dir = os.path.split(os.path.realpath(__file__))[0]
project_dir = os.path.split(os.path.split(os.path.split(exec_dir)[0])[0])[0]
project_dir_sever = '/home/zy/Project/DL-NC'
exec_dir_sever = exec_dir.replace(project_dir, project_dir_sever)
sys.path.append(project_dir)

Optimal_gens = exec_dir + '/Optimal_gens.pkl'

ray_cluster_address = 'auto'

exec_env = '''
from brian2 import *
import warnings
try:
    clear_cache('cython')
except:
    pass
warnings.filterwarnings("ignore")
BrianLogger.log_level_error()
prefs.codegen.target = "numpy"
start_scope()
'''

exec_var = open(os.path.join(exec_dir, "src/config.py")).read()

# -------get numpy random state------------
seed = 100
np.random.seed(seed)
np_state = np.random.get_state()

# -----simulation overall setting-------
core = 60
