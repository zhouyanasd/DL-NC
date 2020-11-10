from multiprocessing import Manager,Queue, Pool
import multiprocessing
import numpy as np
import os
from functools import partial

x = 1
y = 2
z = 0
k = 0
r = np.random.rand()

# test the parallel share memory
def add_sp(a, b, c):
    name = multiprocessing.current_process().name
    c = a + b
    print(name, os.getpid(),id(a), a)
    return c


# test the parallel share memory
def add_s(q, qq, a, b, c):
    name = multiprocessing.current_process().name
    c = a + b
    print(name, os.getpid(),id(q), q.get(), id(c), c)
    qq.put(np.array([c, c]))
    return c


if __name__ == '__main__':
    tries = 4
    pools = 3
    p = Pool(pools)

    q = Manager().Queue()
    qq = Manager().Queue()
    q.put(np.array([100, 200]))
    q.put(np.array([300, 400]))
    q.put(np.array([500, 600]))


    result = p.starmap(partial(add_s,q,qq), zip([x + i for i in range(tries)], [y] * tries, [z] * tries))
    # result = p.starmap(partial(add_sp, x), zip([y + i for i in range(tries)],  [z] * tries))
    print('-----')
    while True:
        if not qq.empty():
            print(result, id(qq), qq.get())
        else:
            break
    print(id(x), x)
    print(result)