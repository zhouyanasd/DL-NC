# ----------------------------------------
# Logistic Regression
# Gradient descent
# multi-class classification(independent)
# ----------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the softmax function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost(X, Y, P):
    left = np.multiply(Y.T, np.log(sigmoid(X.dot(P))))
    right = np.multiply((1 - Y).T, np.log(1 - sigmoid(X.dot(P))))
    return -np.sum((left + right), axis=0) / (len(Y.T))

def train(X, Y, P, rate = 0.0001,theta = 1e-2):
    time = 0
    while (cost(X,Y,P) > theta).all():
        time+=1
        P = P + X.T.dot(Y.T - sigmoid(X.dot(P))) * rate
    print (time, cost(X,Y,P))
    return P

def predict_logistic(results):
    labels = (results>0.5).astype(int).T
    return labels

def prepare_Y(label):
    if np.asarray(label).ndim == 1:
        return np.asarray([label])
    else:
        return np.asarray(label)

X1=np.array([8.19,2.72,6.39,8.71,4.7,9.66,8.78,7.25,2.69,2.65,7.23,2.35,2.26,6.98,2.65])
X2=np.array([7.01,2.78,6.47,6.71,4.1,8.23,4.05,6.36,3.25,3.58,9.36,3.35,4.24,6.21,2.41])
label=np.array([[1,0,1,1,0,1,1,1,0,0,1,0,0,1,0],
                [1,0,1,1,1,1,1,1,0,0,1,0,0,1,0]])

one = np.ones((len(X1), 1))  # len(x)得到数据量 bis
X1 = X1.reshape(X1.shape[0], 1)
X2 = X2.reshape(X2.shape[0], 1)
X = np.hstack((X1, X2, one))
Y = prepare_Y(label)
P = np.random.rand(3,2)

P = train(X, Y, P)
label_predict = predict_logistic(sigmoid(X.dot(P)))

score = [accuracy_score(i,j) for i,j in zip(label,label_predict)]
print("The accruacy socre is " + str(score))


#----plot-------
xs = np.linspace(0, 10, 20)
ys = np.linspace(0, 10, 20)
X_p, Y_p = np.meshgrid(xs, ys)
line = []
for i in range(len(Y)):
    line.append(P[0][i]*X_p+P[1][i]*Y_p+P[2][i])
Z_p = sigmoid(np.array(line))

for i in range(len(Y)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, label[i],color="red",label="Sample Point",linewidth=3)
    ax.plot_surface(X_p,Y_p,Z_p[i],cmap=cm.coolwarm,alpha = 0.5,rstride=1,cstride= 1,linewidth = 0.1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    print('阈值为0.5的等高线（决策区域）：')
    fig2 = plt.figure()
    plt.scatter(X1, X2, c=label[i], cmap=plt.cm.Paired)
    plt.contour(X_p, Y_p, Z_p[i] , [0.5], colors = 'black', linewidth = 0.5)
plt.show()