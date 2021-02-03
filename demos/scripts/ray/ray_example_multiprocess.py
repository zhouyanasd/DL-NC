# from multiprocessing import Queue, Pool
# import multiprocessing
import numpy as np
import os
import ray

ray.init('auto')

x = 1
y = 2
z = 0
r = np.random.rand()

ob_r = ray.put(r)

@ray.remote
def add_ray(a, b, c):
    print(a,b,c)
    print(r == ray.get(ob_r))
    return a+b+c


data_list = ray.get([add_ray.remote(x+k, y, z) for k in range(4)])

print(data_list)



# q = Queue()
# q.put(np.array([100, 200]))
# q.put(np.array([300, 400]))
# q.put(np.array([500, 600]))


# # test the parallel share memory
# def add_s(a, b, c):
#     name = multiprocessing.current_process().name
#     print(name, os.getpid())
#     c = a + b
#     print(id(q), q.get(), id(c), c)
#     q.put(np.array([c, c]))
#     return c
#
#
# # test the parallel object copy
# def add(a, b, c):
#     c = a + b
#     print(os.getpid(), id(c))
#     k = [c]
#     print(k, id(k))
#     return c
#
#
# # to test the parallel get state and reshape
# def add_np(a, b, c):
#     c = a + b
#     return np.array([[a + r, b, c],
#                      [a, b + r, c]]).T


# if __name__ == '__main__':
#     tries = 3
#     p = Pool(tries)
#     result = p.starmap(add_s, zip([x + i for i in range(tries)], [y] * tries, [z] * tries))
#
#     while True:
#         if not q.empty():
#             print(result, id(q), q.get())
#         else:
#             break
#     print(result, id(k), k)
#     # print(np.asarray(result_1).reshape(-1,2).T)
#     print('-----')
#     print(os.getpid(), add(x, y, z))