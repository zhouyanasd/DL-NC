###最小二乘法试验###
import numpy as np
from scipy.optimize import leastsq

###采样点(Xi,Yi)###
Xi=np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
Yi=np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])
Zi=np.array([1,0,1,1,0,1,1])

Data = [Xi,Yi]

###需要拟合的函数func及误差error###
def error(p,y, args):
    f = 0
    for i in range (len(args)):
        f += p[i]*args[i]
    return f-y

# print(error([1,2],2,1,2))


#TEST
p0=[100,100,2]
#print( error(p0,Xi,Yi) )

###主函数从此开始###
Para=leastsq(error,p0,args=(Zi,Data)) #把error函数中除了p以外的参数打包到args中
k1,k2,b=Para[0]
print(Para)
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