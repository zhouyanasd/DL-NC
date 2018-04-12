# ----------------------------------------
# Softmax
# ----------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the softmax function
def softmax(z):
    return np.array([(np.exp(i) / np.sum(np.exp(i))) for i in z])

def cost(X, Y, P):
    temp= np.log(softmax(X.dot(P)))[Y.T == 1]
    return -np.sum(temp) / (len(Y.T))

def train(X, Y, P, rate = 0.0001,theta = 1e-1):
    time = 0
    while cost(X,Y,P) > theta:
        time+=1
        P = P + X.T.dot(Y.T - softmax(X.dot(P))) * rate
    print (time, cost(X,Y,P))
    return P

def predict(results):
    labels = []
    for result in results:
        labels.append(np.argmax(result))
    return labels

def label_to_obj(label, obj):
    temp = []
    for a in label:
        if a == obj:
            temp.append(1)
        else:
            temp.append(0)
    return np.asarray(temp)

def one_versus_the_rest(label, *args, **kwargs):
    obj = []
    for i in args:
        temp = label_to_obj(label, i)
        obj.append(temp)
    try:
         for i in kwargs['selected']:
            temp = label_to_obj(label, i)
            obj.append(temp)
    except KeyError:
        pass
    return np.asarray(obj)

X1=np.array([8.19,2.72,6.39,8.71,0.7,9.66,8.78,2.25,2.69,2.65,7.23,2.35,2.26,6.98,2.65])
X2=np.array([7.01,8.78,6.47,6.71,6.1,8.23,4.05,6.36,3.25,3.58,9.36,3.35,4.24,6.21,9.41])
label = np.array([0,2,0,0,2,0,0,2,1,1,0,1,1,0,2])

one = np.ones((len(X1), 1))  # len(x)得到数据量 bis
X1 = X1.reshape(X1.shape[0], 1)
X2 = X2.reshape(X2.shape[0], 1)
X = np.hstack((X1, X2, one))
Y = one_versus_the_rest(label, selected=np.arange(3))
P = np.random.rand(3,3)

P = train(X, Y, P)
label_predict = predict(softmax(X.dot(P)))

score = accuracy_score(label, label_predict)
print("The accruacy socre is " + str(score))

#----plot----
xs = np.linspace(0, 10, 20)
ys = np.linspace(0, 10, 20)
X_p, Y_p = np.meshgrid(xs, ys)
line = []
for i in range(3):
    line.append(P[0][i]*X_p+P[1][i]*Y_p+P[2][i])
Z_p = softmax(np.array(line).reshape(3,400).T).T.reshape(3,20,20)

print('拟合面：')
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
for i in range(3):
    ax.scatter(X1, X2, Y[i],label="Sample Point",linewidth=3)
    ax.plot_surface(X_p,Y_p,Z_p[i],cmap=cm.coolwarm,alpha = 0.5,rstride=1,cstride= 1,linewidth = 0.1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

print('阈值为0.5的等高线（决策区域）：')
fig2 = plt.figure()
plt.scatter(X1, X2, c=label, cmap=plt.cm.Paired)
for i in range(3):
    plt.contour(X_p, Y_p, Z_p[i], [0.5], colors = 'black', linewidth = 0.5)
plt.show()
