from multiprocessing import Queue, Pool
import multiprocessing
import numpy as np
import os

x = 1
y = 2
z = 0
k = 0
r = np.random.rand()

q = Queue()
q.put(np.array([100, 200]))
q.put(np.array([300, 400]))
q.put(np.array([500, 600]))


# test the parallel share memory
def add_s(a, b, c):
    name = multiprocessing.current_process().name
    print(name, os.getpid())
    c = a + b
    print(id(q), q.get(), id(c), c)
    q.put(np.array([c, c]))
    return c


# test the parallel object copy
def add(a, b, c):
    c = a + b
    print(os.getpid(), id(c))
    k = [c]
    print(k, id(k))
    return c


# to test the parallel get state and reshape
def add_np(a, b, c):
    c = a + b
    return np.array([[a + r, b, c],
                     [a, b + r, c]]).T


if __name__ == '__main__':
    tries = 3
    p = Pool(tries)
    result = p.starmap(add_s, zip([x + i for i in range(tries)], [y] * tries, [z] * tries))

    while True:
        if not q.empty():
            print(result, id(q), q.get())
        else:
            break
    print(result, id(k), k)
    # print(np.asarray(result_1).reshape(-1,2).T)
    print('-----')
    print(os.getpid(), add(x, y, z))




# from multiprocessing import Process, Queue
# import time
#
#
# def reader(queue):
#     while True:
#         msg = queue.get()  # Read from the queue and do nothing
#         if (msg == 'DONE'):
#             break
#
#
# def writer(count, queue):
#     for ii in range(0, count):
#         queue.put(ii)  # Write 'count' numbers into the queue
#     queue.put('DONE')
#
#
# if __name__ == '__main__':
#     for count in [10 ** 4, 10 ** 5, 10 ** 6]:
#         queue = Queue()  # reader() reads from queue
#         # writer() writes to queue
#         reader_p = Process(target=reader, args=((queue),))
#         reader_p.daemon = True
#         reader_p.start()  # Launch the reader process
#
#         _start = time.time()
#         writer(count, queue)  # Send a lot of stuff to reader()
#         reader_p.join()  # Wait for the reader to finish
#         print ("Sending %s numbers to Queue() took %s seconds" % (count,
#                                                            (time.time() - _start)))



# from multiprocessing import Process, Queue, Pool
# import multiprocessing
# import os, time, random
#
#
# # 写数据进程执行的代码:
# def write(q, lock):
#     lock.acquire()  # 加上锁
#     for value in ['A', 'B', 'C']:
#         print ('Put %s to queue...' % value)
#         q.put(value)
#     lock.release()  # 释放锁
#
#
# # 读数据进程执行的代码:
# def read(q):
#     while True:
#         if not q.empty():
#             value = q.get(False)
#             print ('Get %s from queue.' % value)
#             time.sleep(random.random())
#         else:
#             break
#
#
# if __name__ == '__main__':
#     manager = multiprocessing.Manager()
#     # 父进程创建Queue，并传给各个子进程：
#     q = manager.Queue()
#     lock = manager.Lock()  # 初始化一把锁
#     p = Pool()
#     pw = p.apply_async(write, args=(q, lock))
#     pr = p.apply_async(read, args=(q,))
#     p.close()
#     p.join()
#
#     print('所有数据都写入并且读完')