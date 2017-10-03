# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, coordinates):
    totalError = 0
    for i in range(0, len(coordinates)):
        x = coordinates[i][0]
        y = coordinates[i][1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(coordinates))
# example
compute_error = compute_error_for_line_given_points(1, 2, [[3,6],[6,9],[12,18]])
print(compute_error)




current_x = 0.5 # the algorithm starts at x=0.5
learning_rate = 0.01 # step size multiplier
num_iterations = 60 # the number of times to train the function

#the derivative of the error function (x**4 = the power of 4 or x^4)
def slope_at_given_x_value(x):
   return 5 * x**4 - 6 * x**2

# Move X to the right or left depending on the slope of the error function
for i in range(num_iterations):
   previous_x = current_x
   current_x += -learning_rate * slope_at_given_x_value(previous_x)
   print(previous_x)

print("The local minimum occurs at %f" % current_x)




from random import choice
from numpy import array, dot, random

_or_0 = lambda x: 0 if x < 0 else 1
training_data = [(array([0, 0, 1]), 0),
                 (array([0, 1, 1]), 1),
                 (array([1, 0, 1]), 1),
                 (array([1, 1, 1]), 1), ]
weights = random.rand(3)
errors = []
learning_rate = 0.2
num_iterations = 100

for i in range(num_iterations):
    input, truth = choice(training_data)
    result = dot(weights, input)
    error = truth - _or_0(result)
    errors.append(error)
    weights += learning_rate * error * input

for x, _ in training_data:
    result = dot(x, weights)
    print("{}: {} -> {}".format(input[:2], result, _or_0(result)))





import numpy as np

X_XOR = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y_truth = np.array([[0], [1], [1], [0]])

np.random.seed(1)
syn_0 = 2 * np.random.random((3, 4)) - 1
syn_1 = 2 * np.random.random((4, 1)) - 1


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


for j in range(60000):
    layer_1 = sigmoid(np.dot(X_XOR, syn_0))
    layer_2 = sigmoid(np.dot(layer_1, syn_1))
    error = layer_2 - y_truth
    layer_2_delta = error * sigmoid_output_to_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(syn_1.T)
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    syn_1 -= layer_1.T.dot(layer_2_delta)
    syn_0 -= X_XOR.T.dot(layer_1_delta)

print("Output After Training: \n", layer_2)