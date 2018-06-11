from multiprocessing import Queue,Pool
import numpy as np

x = 1
y = 2
z = 0
k = 0
r = np.random.rand()

def add(a,b,c):
    c = a+b
    print(id(c))
    k = [c]
    print(k, id(k))
    return c

def add_np(a,b,c):
    c = a+b
    return np.array([[a+r,b,c],
                     [a,b+r,c]]).T

if __name__ == '__main__':
    tries = 3
    p = Pool(tries)
    result_1 = p.starmap(add_np, zip([x+i for i in range(tries)],[y]*tries, [z]*tries))
    result_2 = p.starmap(add_np, zip([x + i+10 for i in range(tries)], [y] * tries, [z] * tries))
    print(result_1,id(k))
    print(result_2, id(k))
    print(np.asarray(result_1).reshape(-1,2).T)
    print('-----')
    print(add(x,y,z))
