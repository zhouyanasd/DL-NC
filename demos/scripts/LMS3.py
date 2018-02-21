import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def optimal(A, b):
    B = A.T.dot(b)
    AA = np.linalg.inv(A.T.dot(A))  # 求A.T.dot(A)的逆
    P = AA.dot(B)
    print(P)
    return P

# # x的个数决定了样本量
# x = np.arange(-1, 1, 0.02)
# # y为理想函数
# y = 2 * np.sin(x * 2.3) + 0.5 * x ** 3
# y_0 = 1.5 * np.sin(x * 3) + 0.5 * x ** 6
# # y1为离散的拟合数据
# y1 = y + 0.5 * (np.random.rand(len(x)) - 0.5)
#
# y = y.reshape(y.shape[0], 1)
# y_0 = y_0.reshape(y.shape[0], 1)
# Y = np.hstack((y, y_0))
# ##################################
# # 主要程序
# one = np.ones((len(x), 1))  # len(x)得到数据量 bis
# x = x.reshape(x.shape[0], 1)
# A = np.hstack((x, one))  # 两个100x1列向量合并成100x2,(100, 1) (100,1 ) (100, 2)
# C = y1.reshape(y1.shape[0], 1)
#
#
# # 等同于C=y1.reshape(100,1)
# # 虽然知道y1的个数为100但是程序中不应该出现人工读取的数据

#
# # 求得的[a,b]=P=[[  2.88778507e+00] [ -1.40062271e-04]]
# yy = optimal(A, Y)
# # yy=P[0]*x+P[1]
# #################################
# plt.plot(x, y, color='g', linestyle='-', marker='', label=u'理想曲线')
# plt.plot(x, y_0, color='g', linestyle='-', marker='', label=u'理想曲线')
# plt.plot(x, y1, color='m', linestyle='', marker='o', label=u'拟合数据')
# plt.plot(x, yy, color='b', linestyle='-', marker='.', label=u"拟合曲线")
# # 把拟合的曲线在这里画出来
# plt.legend(loc='upper left')
# plt.show()



Xi=np.array([8.19,2.72,6.39,8.71,0.7,9.66,8.78,2.25,2.69,2.65,7.23,2.35,2.26,6.98,2.65])
Yi=np.array([7.01,8.78,6.47,6.71,6.1,8.23,4.05,6.36,3.25,3.58,9.36,3.35,4.24,6.21,9.41])

Zi=np.array([[1,0,1,1,0,1,1,0,0,0,1,0,0,1,0],
             [0,0,0,0,0,0,0,0,1,1,0,1,1,0,0],
             [0,1,0,0,1,0,0,1,0,0,0,0,0,0,1]])

one = np.ones((len(Xi), 1))  # len(x)得到数据量 bis
Xi = Xi.reshape(Xi.shape[0], 1)
Yi = Yi.reshape(Yi.shape[0], 1)
A = np.hstack((Xi, Yi, one))  # 两个100x1列向量合并成100x2,(100, 1) (100,1 ) (100, 2)
# C = Zi.reshape(Zi.shape[0], 1)

P = optimal(A, Zi.T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xi, Yi, Zi[0],color="red",label="Sample Point",linewidth=3)
ax.scatter(Xi, Yi, Zi[1],color="red",label="Sample Point",linewidth=3)
ax.scatter(Xi, Yi, A.dot(P).T[0],color="blue",label="Sample Point",linewidth=1)
ax.scatter(Xi, Yi, A.dot(P).T[1],color="blue",label="Sample Point",linewidth=1)

xs = np.linspace(0, 10, 20)
ys = np.linspace(0, 10, 20)
X, Y = np.meshgrid(xs, ys)
Z0 = P[0][0]*X+P[1][0]*Y+P[2][0]
Z1 = P[0][1]*X+P[1][1]*Y+P[2][1]
Z2 = P[0][2]*X+P[1][2]*Y+P[2][2]

ax.plot_surface(X,Y,Z0,cmap=cm.coolwarm,alpha = 0.5,rstride=1,cstride= 1,linewidth = 0.1)
ax.plot_surface(X,Y,Z1,cmap=cm.coolwarm,alpha = 0.5,rstride=1,cstride= 1,linewidth = 0.1)
ax.plot_surface(X,Y,Z2,cmap=cm.coolwarm,alpha = 0.5,rstride=1,cstride= 1,linewidth = 0.1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

fig2 = plt.figure()
Z = np.array([0,2,0,0,2,0,0,2,1,1,0,1,1,0,2])

plt.scatter(Xi, Yi, c=Z, cmap=plt.cm.Paired)
plt.contour(X, Y, Z0, [0.5], colors = 'black', linewidth = 0.5)
plt.contour(X, Y, Z1, [0.5], colors = 'black', linewidth = 0.5)
plt.contour(X, Y, Z2, [0.5], colors = 'black', linewidth = 0.5)
plt.show()


#-------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(-1, 1, 0.02)
# y = ((x * x - 1) ** 3 + 1) * (np.cos(x * 2) + 0.6 * np.sin(x * 1.3))
#
# y1 = y + (np.random.rand(len(x)) - 0.5)
#
# ##################################
# ### 核心程序
# # 使用函数y=ax^3+bx^2+cx+d对离散点进行拟合，最高次方需要便于修改，所以不能全部列举，需要使用循环
# # A矩阵
# m = []
# for i in range(7):  # 这里选的最高次为x^7的多项式
#     a = x ** (i)
#     m.append(a)
# A = np.array(m).T
# b = y1.reshape(y1.shape[0], 1)
#
#
# ##################################
#
# def projection(A, b):
#     AA = A.T.dot(A)  # A乘以A转置
#     w = np.linalg.inv(AA).dot(A.T).dot(b)
#     print(w)  # w=[[-0.03027851][ 0.1995869 ] [ 2.43887827] [ 1.28426472][-5.60888682] [-0.98754851][ 2.78427031]]
#     return A.dot(w)
#
#
# yw = projection(A, b)
# yw.shape = (yw.shape[0],)
#
# plt.plot(x, y, color='g', linestyle='-', marker='', label=u"理想曲线")
# plt.plot(x, y1, color='m', linestyle='', marker='o', label=u"已知数据点")
# plt.plot(x, yw, color='r', linestyle='', marker='.', label=u"拟合曲线")
# plt.legend(loc='upper left')
# plt.show()