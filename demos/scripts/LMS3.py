###最小二乘法试验###
import numpy as np
from scipy.optimize import leastsq

###采样点(Xi,Yi)###
# Xi=np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78,1.25,2.69,8.65,7.23,6.35,8.26,6.98,2.65])
# Yi=np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05,2.36,7.25,6.58,9.36,8.35,7.24,5.21,2.41])

Xi=np.array([8.19,2.72,6.39,8.71,4.7,9.66,8.78,7.25,2.69,2.65,7.23,2.35,2.26,6.98,2.65])
Yi=np.array([7.01,2.78,6.47,6.71,4.1,8.23,4.05,6.36,3.25,3.58,9.36,3.35,4.24,6.21,2.41])

# Xi=np.array([8,2,8,8,2,8,8,8,2,2,8,2,2,8,2])
# Yi=np.array([8,2,8,8,2,8,8,8,2,2,8,2,2,8,2])
Zi=np.array([[1,0,1,1,0,1,1,1,0,0,1,0,0,1,0],
             [1,0,1,1,0,1,1,1,0,0,1,0,0,1,0]])

Data = [Xi,Yi]

###需要拟合的函数func及误差error###
def error(p,y, args):
    l = p.shape[0]
    f = p[:,l-1]
    print(y,len(args) )
    for i in range (len(args)):
        f += p[i]*args[i]
    return f-y

# print(error([1,2],2,1,2))


#TEST
p0=np.array([[100,100,0.1],
             [100,100,0.1]])
#print( error(p0,Xi,Yi) )

###主函数从此开始###
Para=leastsq(error,p0,args=(Zi,Data)) #把error函数中除了p以外的参数打包到args中
k1,k2,b=Para[0]
print(Para)
print("k1=",k1,"k2=",k2,'\n',"b=",b)

# ###绘图，看拟合效果###

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xi, Yi, Zi,color="red",label="Sample Point",linewidth=3)

xs = np.linspace(0, 10, 20)
ys = np.linspace(0, 10, 20)
X, Y = np.meshgrid(xs, ys)
Z = k1*X+k2*Y+b

ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,alpha = 0.5,rstride=1,cstride= 1,linewidth = 0.1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
