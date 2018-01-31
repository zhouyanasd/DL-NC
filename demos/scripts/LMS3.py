import numpy as np
import matplotlib.pyplot as plt

# x的个数决定了样本量
x = np.arange(-1, 1, 0.02)
# y为理想函数
y = 2 * np.sin(x * 2.3) + 0.5 * x ** 3
y_0 = 1.5 * np.sin(x * 3) + 0.5 * x ** 6
# y1为离散的拟合数据
y1 = y + 0.5 * (np.random.rand(len(x)) - 0.5)

y = y.reshape(y.shape[0], 1)
y_0 = y_0.reshape(y.shape[0], 1)
Y = np.hstack((y, y_0))
##################################
# 主要程序
one = np.ones((len(x), 1))  # len(x)得到数据量 bis
x = x.reshape(x.shape[0], 1)
A = np.hstack((x, one))  # 两个100x1列向量合并成100x2,(100, 1) (100,1 ) (100, 2)
C = y1.reshape(y1.shape[0], 1)


# 等同于C=y1.reshape(100,1)
# 虽然知道y1的个数为100但是程序中不应该出现人工读取的数据

def optimal(A, b):
    B = A.T.dot(b)
    AA = np.linalg.inv(A.T.dot(A))  # 求A.T.dot(A)的逆
    P = AA.dot(B)
    print(P)
    return A.dot(P)


# 求得的[a,b]=P=[[  2.88778507e+00] [ -1.40062271e-04]]
yy = optimal(A, Y)
# yy=P[0]*x+P[1]
#################################
plt.plot(x, y, color='g', linestyle='-', marker='', label=u'理想曲线')
plt.plot(x, y_0, color='g', linestyle='-', marker='', label=u'理想曲线')
plt.plot(x, y1, color='m', linestyle='', marker='o', label=u'拟合数据')
plt.plot(x, yy, color='b', linestyle='-', marker='.', label=u"拟合曲线")
# 把拟合的曲线在这里画出来
plt.legend(loc='upper left')
plt.show()



Xi=np.array([8.19,2.72,6.39,8.71,4.7,9.66,8.78,7.25,2.69,2.65,7.23,2.35,2.26,6.98,2.65])
Yi=np.array([7.01,2.78,6.47,6.71,4.1,8.23,4.05,6.36,3.25,3.58,9.36,3.35,4.24,6.21,2.41])

Zi=np.array([1,0,1,1,0,1,1,1,0,0,1,0,0,1,0])

