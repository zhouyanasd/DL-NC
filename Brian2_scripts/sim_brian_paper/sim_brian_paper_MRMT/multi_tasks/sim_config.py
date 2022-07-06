import os, sys
import numpy as np

# -----save state ------
is_save_state = False

# -------path settings------------
exec_dir = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
project_dir = os.path.split(os.path.split(os.path.split(exec_dir)[0])[0])[0]
project_dir_sever = '/home/zy/Project/DL-NC'
exec_dir_sever = exec_dir.replace(project_dir, project_dir_sever)
sys.path.append(project_dir)

Optimal_gens = exec_dir + '/Optimal_gens_'
Optimal_state = exec_dir + '/Optimal_state_'

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
seeds = 100
np.random.seed(seeds)
np_state = np.random.get_state()

# -----simulation overall setting-------
cores = 60

# --------------------------------------