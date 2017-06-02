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
size_l =[]
while True:
    lines = s.readline()
    i += 1
    if not lines:
        print(size_l)
        break
    if lines == '\n':
        i -= 1
        size_l.append(i)
        continue

data_l=np.loadtxt("../Data/jv/size.txt", delimiter=None).astype(int)[1]

data_l = np.cumsum(data_l)
print(data_l)

label = []
for i in range(len(data_l)):
    for j in range(data_l[i]):
        for l in range(size_l[j]):
            if i == 1:
                label.append(1)
            else:
                label.append(0)

print(label)