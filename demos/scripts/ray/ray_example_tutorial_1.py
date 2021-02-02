import time

def do_some_work(x):
    time.sleep(1) # Replace this with work you need to do.
    return x

start = time.time()
results = [do_some_work(x) for x in range(4)]
print("duration =", time.time() - start)
print("results = ", results)

# ------

import time
import ray

ray.init(num_cpus = 4) # Specify this system has 4 CPUs.

@ray.remote
def do_some_work(x):
    time.sleep(1) # Replace this is with work you need to do.
    return x

start = time.time()
results = [do_some_work.remote(x) for x in range(4)]
# results = [ray.get(do_some_work.remote(x)) for x in range(4)]
#results = ray.get([do_some_work.remote(x) for x in range(4)])
print("duration =", time.time() - start)
print("results = ", results)

# ----

import time

def tiny_work(x):
    time.sleep(0.0001) # Replace this with work you need to do.
    return x

start = time.time()
results = [tiny_work(x) for x in range(100000)]
print("duration =", time.time() - start)


# ----
import time
import ray

ray.init(num_cpus = 4)

@ray.remote
def tiny_work(x):
    time.sleep(0.0001) # Replace this is with work you need to do.
    return x

start = time.time()
result_ids = [tiny_work.remote(x) for x in range(100000)]
results = ray.get(result_ids)
print("duration =", time.time() - start)

#----
import time
import ray

ray.init(num_cpus = 4)

def tiny_work(x):
    time.sleep(0.0001) # replace this is with work you need to do
    return x

@ray.remote
def mega_work(start, end):
    return [tiny_work(x) for x in range(start, end)]

start = time.time()
result_ids = []
[result_ids.append(mega_work.remote(x*1000, (x+1)*1000)) for x in range(100)]
results = ray.get(result_ids)
print("duration =", time.time() - start)

#----
@ray.remote
def no_work(x):
    return x

start = time.time()
num_calls = 1000
[ray.get(no_work.remote(x)) for x in range(num_calls)]
print("per task overhead (ms) =", (time.time() - start)*1000/num_calls)

#----
import time
import numpy as np
import ray

ray.init(num_cpus = 4)

@ray.remote
def no_work(a):
    return

start = time.time()
a = np.zeros((5000, 5000))
result_ids = [no_work.remote(a) for x in range(10)]
results = ray.get(result_ids)
print("duration =", time.time() - start)

#----

import time
import numpy as np
import ray

ray.init(num_cpus = 4)

@ray.remote
def no_work(a):
    return

start = time.time()
a_id = ray.put(np.zeros((5000, 5000)))
result_ids = [no_work.remote(a_id) for x in range(10)]
results = ray.get(result_ids)
print("duration =", time.time() - start)

#----
import time
import random
import ray

ray.init(num_cpus = 4)

@ray.remote
def do_some_work(x):
    time.sleep(random.uniform(0, 4)) # Replace this with work you need to do.
    return x

def process_results(results):
    sum = 0
    for x in results:
        time.sleep(1) # Replace this with some processing code.
        sum += x
    return sum

start = time.time()
data_list = ray.get([do_some_work.remote(x) for x in range(4)])
sum = process_results(data_list)
print("duration =", time.time() - start, "\nresult = ", sum)

#----
import time
import random
import ray

ray.init(num_cpus = 4)

@ray.remote
def do_some_work(x):
    time.sleep(random.uniform(0, 4)) # Replace this is with work you need to do.
    return x

def process_incremental(sum, result):
    time.sleep(1) # Replace this with some processing code.
    return sum + result

start = time.time()
result_ids = [do_some_work.remote(x) for x in range(4)]
sum = 0
while len(result_ids):
    done_id, result_ids = ray.wait(result_ids)
    print(done_id, result_ids)
    sum = process_incremental(sum, ray.get(done_id[0]))
print("duration =", time.time() - start, "\nresult = ", sum)