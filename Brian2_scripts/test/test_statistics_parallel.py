from multiprocessing import Pool

from math import hypot
from random import random
import time


def test(tries):
    return sum(hypot(random(), random()) < 1 for _ in range(tries))


def calcPi(kernel,nbFutures, tries):
    ts = time.time()
    p = Pool(kernel)
    result = p.map(test, [tries] * nbFutures)
    ret = 4. * sum(result) / float(nbFutures * tries)
    span = time.time() - ts
    print ("time spend ", span)
    return ret


if __name__ == '__main__':
    print("pi = {}".format(calcPi(4, 3000, 4000)))

#-------------------------------------------
# coding:gbk

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
#     time.sleep(0.5)
#     pr = p.apply_async(read, args=(q,))
#     p.close()
#     p.join()
#
#     print('所有数据都写入并且读完')

#--------------------------------
#coding:gbk

# from multiprocessing import Process, Queue
# import os, time, random
#
# # 写数据进程执行的代码:
# def write(q):
#     for value in ['A', 'B', 'C']:
#         print ('Put %s to queue...' % value)
#         q.put(value)
#         time.sleep(random.random())
#
# # 读数据进程执行的代码:
# def read(q):
#     while True:
#         if not q.empty():
#             value = q.get(True)
#             print ('Get %s from queue.' % value)
#             time.sleep(random.random())
#         else:
#             break
#
# if __name__=='__main__':
#     # 父进程创建Queue，并传给各个子进程：
#     q = Queue()
#     pw = Process(target=write, args=(q,))
#     pr = Process(target=read, args=(q,))
#     # 启动子进程pw，写入:
#     pw.start()
#     # 等待pw结束:
#     pw.join()
#     # 启动子进程pr，读取:
#     pr.start()
#     pr.join()
#     # pr进程里是死循环，无法等待其结束，只能强行终止:
#     print ('所有数据都写入并且读完')