###最小二乘法试验###
import numpy as np
from scipy.optimize import leastsq

###采样点(Xi,Yi)###
Xi=np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
Yi=np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])
Zi=np.array([3.01,1.78,4.47,5.71,1.1,2.23,1.05])

###需要拟合的函数func及误差error###
def func(p,x,y):
    k1,k2,b=p
    return k1*x+k2*y+b

def error(p,x,y,z,s):
    print (s)
    return func(p,x,y)-z #x、y都是列表，故返回值也是个列表

#TEST
p0=[100,100,2]
#print( error(p0,Xi,Yi) )

###主函数从此开始###
s="Test the number of iteration" #试验最小二乘法函数leastsq得调用几次error函数才能找到使得均方误差之和最小的k、b
Para=leastsq(error,p0,args=(Xi,Yi,Zi,s)) #把error函数中除了p以外的参数打包到args中
k1,k2,b=Para[0]
print("k1=",k1,"k2=",k2,'\n',"b=",b)

# ###绘图，看拟合效果###
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8,6))
# plt.scatter(Xi,Yi,Zi,color="red",label="Sample Point",linewidth=3) #画样本点
# x=np.linspace(0,10,1000)
# y=np.linspace(0,10,1000)
# z=k1*x+k2*y+b
# plt.plot(x,y,z,color="orange",label="Fitting Line",linewidth=2) #画拟合直线
# plt.legend()
# plt.show()