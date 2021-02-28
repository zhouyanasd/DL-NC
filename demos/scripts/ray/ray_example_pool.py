from ray.util.multiprocessing import Pool
import ray

def f(index):
    print(ray._private.services.get_node_ip_address())
    return index

pool = Pool(processes=10, ray_address="auto", maxtasksperchild = 1)
for result in pool.map(f, range(100)):
    print(result)