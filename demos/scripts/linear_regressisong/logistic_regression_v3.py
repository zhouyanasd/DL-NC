# ----------------------------------------
# Softmax
# ----------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score

# Define the softmax function
def softmax(z):
    return np.array([(np.exp(i) / np.sum(np.exp(i))) for i in z])

def train(X, Y, P):
    a = 0.0001
    max_iteration = 10000
    time = 0
    while time < max_iteration:
        time += 1
        P = P + X.T.dot(Y.T - softmax(X.dot(P))) * a
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

