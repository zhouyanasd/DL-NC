# _*_ coding: utf-8 _*_
# 作者: yhao
# 博客: http://blog.csdn.net/yhao2014
# 邮箱: yanhao07@sina.com

import numpy as np  # 引入numpy
import pylab as pl
from scipy.optimize import leastsq  # 引入最小二乘函数

n = 9  # 多项式次数


# 目标函数
def real_func(x):
    return np.sin(2 * np.pi * x)


# 多项式函数
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


# 残差函数
def residuals_func(p, y, x):
    ret = fit_func(p, x) - y
    return ret


x = np.linspace(0, 1, 9)  # 随机选择9个点作为x
x_points = np.linspace(0, 1, 1000)  # 画图时需要的连续点

y0 = real_func(x)  # 目标函数
y1 = [np.random.normal(0, 0.1) + y for y in y0]  # 添加正太分布噪声后的函数

p_init = np.random.randn(1,n)  # 随机初始化多项式参数

plsq = leastsq(residuals_func, p_init, args=(y1, x))
print ('Fitting Parameters: ', plsq[0])  # 输出拟合参数

pl.plot(x_points, real_func(x_points), label='real')
pl.plot(x_points, fit_func(plsq[0], x_points), label='fitted curve')
pl.plot(x, y1, 'bo', label='with noise')
pl.legend()
pl.show()