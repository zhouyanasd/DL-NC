from multiprocessing import Queue,Pool

x = 1
y = 2
z = 0
k = 0

def add(a,b,c):
    c = a+b
    print(id(c))
    k = [c]
    print(k, id(k))
    return c

if __name__ == '__main__':
    tries = 3
    p = Pool(tries)
    result = p.starmap(add, zip([x+i for i in range(tries)],[y]*tries, [z]*tries))
    print(result,id(k))
    print('-----')
    print(add(x,y,z))
