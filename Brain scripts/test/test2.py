import numpy as np

def read_from_txt(filename=None):
    """
    read dataset from txt and return np.array
    :param filename: filename
    :return: data (type of np.array)
    """
    with open(filename, 'r') as file_to_read:
        data = []
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            if lines =='\n':
                continue
            data_line = [float(i) for i in lines.split(' ')]
            data.append(data_line)
        data = np.array(data)
        return data

a = np.loadtxt("../Data/jv/train.txt", delimiter=None)
print(a.shape)

s = open("../Data/jv/train.txt",'r')
i = -1
j =[]
while True:
    lines = s.readline()
    i += 1
    if not lines:
        print(j)
        break
    if lines == '\n':
        i -= 1
        j.append(i)
        continue

print(a[19])
print(a[45])