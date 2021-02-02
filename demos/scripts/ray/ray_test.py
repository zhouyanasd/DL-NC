import time
import ray
ray.init('auto')

@ray.remote
def f():
    time.sleep(0.01)
#    return ray._private.services.get_node_ip_address()
    return ray.nodes()

# Get a list of the IP addresses of the nodes that have joined the cluster.
print(ray.get([f.remote() for _ in range(2)]))