from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.core.cluster import Cluster
from Brian2_scripts.sim_brian_paper.sim_brian_paper_CoE.src.ray_config import ray_cluster_one, ray_cluster_two

import time
import numpy as np
import ray, os
from ray.exceptions import RayActorError, WorkerCrashedError, RayError
from ray.util.multiprocessing import Pool


cluster = Cluster(ray_cluster_one)

cluster.start()


def test_run_1():

    @ray.remote(max_retries=1)
    def potentially_fail(failure_probability):
        time.sleep(0.2)
        if np.random.random() < failure_probability:
            os._exit(0)
        return 0

    ray.init('auto')

    while True:
        try:
            # If this task crashes, Ray will retry it up to one additional
            # time. If either of the attempts succeeds, the call to ray.get
            # below will return normally. Otherwise, it will raise an
            # exception.
            ray.get(potentially_fail.remote(0.5))
            print('SUCCESS')
        except WorkerCrashedError as e:
            print('FAILURE: ', e)
            ray.shutdown()
            cluster.restart()
            ray.init('auto')


def test_run_2():

    @ray.remote(max_restarts=5)
    class Actor:
        def __init__(self):
            self.counter = 0

        def increment_and_possibly_fail(self):
            self.counter += 1
            time.sleep(0.2)
            if self.counter == 10:
                os._exit(0)
            return self.counter

    ray.init('auto')
    actor = Actor.remote()

    # The actor will be restarted up to 5 times. After that, methods will
    # always raise a `RayActorError` exception. The actor is restarted by
    # rerunning its constructor. Methods that were sent or executing when the
    # actor died will also raise a `RayActorError` exception.
    for _ in range(100):
        try:
            counter = ray.get(actor.increment_and_possibly_fail.remote())
            print(counter)
        except RayActorError as e:
            print('FAILURE: ', e)
            ray.shutdown()
            cluster.restart()
            ray.init('auto')
            actor = Actor.remote()
            continue


def test_run_3():

    def test_fuck(i):
        time.sleep(0.2)
        if i == 10:
            os._exit(0)
        return i

    while True:
        try:
            pool = Pool(processes=60, ray_address='auto', maxtasksperchild=None)
            result = pool.map(test_fuck, range(100))
            break
        # except RayActorError as e:
        #     print('FAILURE: ', e)
        #     ray.shutdown()
        #     cluster.restart()
        #     continue
        except RayError as e:
            print(e)
            ray.shutdown()
            cluster.restart()
            continue
        except Exception as e:
            print(e)
            continue
        finally:
            print(result)

def test_run_4():

    def f(index):
        print(ray._private.services.get_node_ip_address())
        time.sleep(30)
        return index

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

test_run_1()

# ray.shutdown()

# cluster.stop()