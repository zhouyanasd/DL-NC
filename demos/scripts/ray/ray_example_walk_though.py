import ray, time
from brian2 import *
ray.init('auto')

kkk = 12

# A regular Python function.
def my_function():
    return 1

# By adding the `@ray.remote` decorator, a regular Python function
# becomes a Ray remote function.
@ray.remote
def my_function():
    return 1

# To invoke this remote function, use the `remote` method.
# This will immediately return an object ref (a future) and then create
# a task that will be executed on a worker process.
obj_ref = my_function.remote()

# The result can be retrieved with ``ray.get``.
assert ray.get(obj_ref) == 1

@ray.remote
def slow_function():
    time.sleep(10)
    return 1

# Invocations of Ray remote functions happen in parallel.
# All computation is performed in the background, driven by Ray's internal event loop.
t = time.time()
for _ in range(4):
    # This doesn't block.
    slow_function.remote()
    print(time.time() - t, defaultclock.dt, kkk+_)