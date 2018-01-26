import numpy as np
import struct
import gzip
import matplotlib.pyplot as plt


def readfile():
    with open('../../Data/MNIST_data/train-images.idx3-ubyte','rb') as f1:
        buf1 = f1.read()
    with open('../../Data/MNIST_data/train-labels.idx1-ubyte','rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


def get_image(buf1):
    image_index = 0
    image_index += struct.calcsize('>IIII')
    im = []
    for i in range(9):
        temp = struct.unpack_from('>784B', buf1, image_index) # '>784B'的意思就是用大端法读取784个unsigned byte
        im.append(np.reshape(temp,(28,28)))
        image_index += struct.calcsize('>784B')  # 每次增加784B
    return im


def get_label(buf2): # 得到标签数据
    label_index = 0
    label_index += struct.calcsize('>II')
    return struct.unpack_from('>9B', buf2, label_index)


if __name__ == "__main__":
    image_data, label_data = readfile()
    im = get_image(image_data)
    label = get_label(label_data)

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        title = u"标签对应为："+ str(label[i])
        plt.title(title, fontproperties='SimHei')
        plt.imshow(im[i], cmap='gray')
    plt.show()
