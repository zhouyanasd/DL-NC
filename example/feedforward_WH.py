import numpy as np
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio


def get_WH_data(path = '../Brian2_scripts/Data/WH/WH_TestDataset.mat'):
    data = sio.loadmat(path)
    input_u = data['dataMeas'][0][0][1].T[0]
    output_y = data['dataMeas'][0][0][2].T[0]
    return MinMaxScaler().fit_transform(input_u.reshape(-1,1)).T[0], \
           MinMaxScaler().fit_transform(output_y.reshape(-1, 1)).T[0]


class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(-1, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size,)) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)


# Make up some data
x_data,y_data= get_WH_data()
x_data = x_data[:, np.newaxis]
y_data = y_data[:, np.newaxis]

x1 = x_data[:1382,:]
x2= x_data[1:1383,:]
x3 = x_data[2:1384,:]
x4 = y_data[1:1383,:]

x_data = np.hstack((x1,x2,x3,x4))
y_data = y_data[2:1384,:]

t = np.arange(1,1383)

# dtertermine the inputs dtype
x = T.dmatrix('x')
y = T.dmatrix('y')

# add layer
l1 = Layer(x, 4, 10, T.nnet.sigmoid)
l2 = Layer(l1.outputs, 10, 1, None)

# compute the cost
cost = T.mean(T.square(l2.outputs - y))

# compute the gradients
gW1, gb1, gW2, gb2 = T.grad(cost, [l1.W, l1.b, l2.W, l2.b])

# apply gradient descent
learning_rate = 0.05
train = theano.function(
    inputs=[x, y],
    outputs=cost,
    updates=[
            (l1.W, l1.W - learning_rate * gW1),
             (l1.b, l1.b - learning_rate * gb1),
             (l2.W, l2.W - learning_rate * gW2),
             (l2.b, l2.b - learning_rate * gb2)]
)

# prediction
predict = theano.function(inputs=[x], outputs=l2.outputs)

# plot fake data
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(t,y_data)
plt.ion()
plt.show()

for i in range(2000):
    # training
    err = train(x_data, y_data)
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        print(err)
        # to visualize the result and improvement
        predicted_value = predict(x_data)
        # plot the prediction
        lines = ax.plot(t,predicted_value, 'r-', lw=1)
        plt.pause(1)
    if i == 999:
        fig2 = plt.figure(figsize=(20, 8))
        predicted_value = predict(x_data)
        plt.plot(t, predicted_value, 'r-', lw=1)
        plt.show()


