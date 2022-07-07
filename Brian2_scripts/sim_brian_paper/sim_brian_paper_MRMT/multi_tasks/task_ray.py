# -*- coding: utf-8 -*-
"""
    The fundamental class of network runner for tasks.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

from Brian2_scripts.sim_brian_paper.sim_brian_paper_MRMT.multi_tasks.sim_config import *

from brian2 import *

import ray
from ray.util.multiprocessing import Pool
from ray.exceptions import RayActorError, WorkerCrashedError, RayError


def parallel_run(cluster, fun, data):
    data_list = [x for x in data]
    while True:
        try:
            # ------apply the pool-------
            pool = Pool(processes=core, ray_address=ray_cluster_address, maxtasksperchild=None)
            result = pool.map(fun, data_list)
            # ------close the pool-------
            pool.close()
            pool.join()
            return result
        except (RayActorError, WorkerCrashedError) as e:
            print('restart task: ', e)
            cluster.reconnect(cluster.check_alive())
        except RayError as e:
            print('restart task: ', e)
            ray.shutdown()
            cluster.restart()
        except Exception as e:
            print('restart task: ', e.__class__.__name__, e)
            ray.shutdown()
            cluster.restart()