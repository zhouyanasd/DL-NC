from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core.cluster import Cluster
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.ray_config import ray_cluster_one, ray_cluster_two

import time
import ray
from ray.exceptions import RayActorError, WorkerCrashedError, RayError
from ray.util.multiprocessing import Pool


cluster = Cluster(ray_cluster_one)

cluster.start()

ray.init('auto')

def f(index):
    print(ray._private.services.get_node_ip_address())
    time.sleep(30)
    return index

def test_run():
    while True:
        try:
            pool = Pool(processes=60, ray_address="auto")
            for result in pool.map(f, range(1000000)):
                print(result)
            break
        except (RayActorError, WorkerCrashedError) as r:
            print(r)
            alive = cluster.check_alive()
            cluster.reconnect(alive)
        except RayError as e:
            print(e)
            cluster.restart()
        except Exception as e:
            print(e)
        finally:
            print('unknown error, restart task')

test_run()

ray.shutdown()

cluster.stop()


