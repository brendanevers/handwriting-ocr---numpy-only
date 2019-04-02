import numpy as np
from mnist import MNIST
import sys


def progress_bar(value, cost, endvalue, bar_length=20): # Progress bar for gradient descent
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rIteration is " + str(value) + " cost is " + str(cost)[:5] + " Progress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()


def sigmoid(z):  # Sigmoid function for convenience
    return 1. / (1. + np.exp(-z))


def sigmoid_gradient(z):  # Sigmoid gradient, also for convenience
    return sigmoid(z) * (1. - sigmoid(z))


def bias(X):  # Adds bias neuron, slightly clunky implementation to work with particulars of np.concatenate
    new_front = np.ones((X.shape[0], 1))
    out = np.concatenate((np.transpose(new_front), np.transpose(X)))
    return np.transpose(out)


def rand_initial_weight(l_in, l_out):  # Creates random initial weights for layers
    epsilon = 0.12
    w = np.random.rand(l_out, 1+l_in)*2*epsilon-epsilon
    return w


def y_matrix(y, m, number_range):  # Converts y vector into y matrix for one vs all implementation
    y_mat = np.zeros((m, number_range))
    for i in range(m):
        y_mat[i, y[i]] = 1
    return y_mat


def cost_function(theta_1, theta_2, X, y, input_layer_size, number_range, m, lam):  # Computes the cost function of the neural network
    j = 0
    layer_1 = sigmoid(np.dot(bias(X), np.transpose(theta_1)))
    h = sigmoid(np.dot(bias(layer_1), np.transpose(theta_2)))
    y_mat = y_matrix(y, m, number_range)
    # Add 1e-30 in order to avoid NaN errors due to Python eccentricities (Avoid 0*NaN = NaN)
    for i in range(h[0, :].shape[0]):
        j += -1 / m * (np.matmul(np.transpose(y_mat[:, i]), np.log(h[:, i]+1e-30)) + np.matmul(np.transpose(1 - y_mat[:, i]), np.log(1 - h[:, i]+1e-30)))
    reg = (lam/(2*m)) * (np.sum(np.sum(theta_2[:, 1:] * theta_2[:, 1:])) + np.sum(np.sum(theta_1[:, 1:] * theta_1[:, 1:])))
    j += reg
    return j


def gradient(theta_1, theta_2, X, y, input_layer_size, number_range, m, lam):  # Back propagation algorithm
    y_mat = y_matrix(y, m, number_range)
    a_1 = bias(X)
    z_2 = np.matmul(a_1, np.transpose(theta_1))
    a_2 = bias(sigmoid(z_2))
    z_3 = np.matmul(a_2, np.transpose(theta_2))
    a_3 = sigmoid(z_3)
    del_3 = a_3 - y_mat
    del_2 = np.matmul(del_3, theta_2) * bias(sigmoid_gradient(z_2))
    new_theta_1 = np.matmul(np.transpose(del_2[:, 1:]), a_1) / m
    new_theta_2 = np.matmul(np.transpose(del_3), a_2) / m
    new_theta_1[:, 1:] = new_theta_1[:, 1:] + lam/m * new_theta_1[:, 1:]
    new_theta_2[:, 1:] = new_theta_2[:, 1:] + lam/m * new_theta_2[:, 1:]
    return [new_theta_1, new_theta_2]


def predict(theta_1, theta_2, X):  # Creates prediction vector from inputs
    p = []
    layer_1 = sigmoid(np.matmul(bias(X), np.transpose(theta_1)))
    h = sigmoid(np.matmul(bias(layer_1), np.transpose(theta_2)))
    for i in range(h.shape[0]):
        p.append(np.argmax(h[i]))
    p = np.transpose(np.array(p))
    return p


def accuracy(pred, y): # compares predictions to y vector
    total = 0
    acc = 0
    for i in range(pred.shape[0]):
        if pred[i] == y[i]:
            acc += 1
        total += 1
    return acc/total


if __name__ == '__main__':
    # Parameters for neural network and regularization
    input_layer_size = 784
    hidden_layer_size = 25
    number_range = 10
    lam = 1
    # Import the MNIST sample images and place the first 5000 in arrays
    mndata = MNIST('samples')
    mndata.gz = True
    images, labels = mndata.load_training()
    X = np.asarray(images[:5000])
    y = np.asarray(labels[:5000])
    m = X.shape[0]
    # Create random initial weights
    initial_theta_1 = rand_initial_weight(input_layer_size, hidden_layer_size)
    initial_theta_2 = rand_initial_weight(hidden_layer_size, number_range)
    # Calculate an initial cost and gradient
    cost = cost_function(initial_theta_1, initial_theta_2, X, y, input_layer_size, number_range, m, lam)
    [theta_1_grad, theta_2_grad] = gradient(initial_theta_1, initial_theta_2, X, y, input_layer_size, number_range, m, lam)
    # Perform Gradient Descent
    alpha = 0.1
    descent_depth = 5000
    for i in range(descent_depth):
        progress_bar(i+1, cost, descent_depth, bar_length=20)
        initial_theta_1 -= alpha * theta_1_grad
        initial_theta_2 -= alpha * theta_2_grad
        [theta_1_grad, theta_2_grad] = gradient(initial_theta_1, initial_theta_2, X, y, input_layer_size, number_range, m, lam)
        cost = cost_function(initial_theta_1, initial_theta_2, X, y, input_layer_size, number_range, m, lam)
    # Save new thetas
    np.savetxt("theta_1.csv", initial_theta_1, delimiter=",")
    np.savetxt("theta_2.csv", initial_theta_2, delimiter=",")
    # Test accuracy on test set
    test_X = np.asarray(images[5000:10000])
    test_y = np.asarray(labels[5000:10000])
    pred = predict(initial_theta_1, initial_theta_2, test_X)
    acc = accuracy(pred, test_y) * 100
    print('\n'+'Accuracy is ', acc)

