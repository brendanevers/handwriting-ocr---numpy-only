import numpy as np
from mnist import MNIST
import random


def bias(X):
    return np.insert(1, 1, X)


def sigmoid(z):  # Sigmoid function for convenience
    return 1. / (1. + np.exp(-z))


def predict(theta_1, theta_2, X):  # Creates prediction vector from inputs
    p = []
    layer_1 = sigmoid(np.matmul(bias(X), np.transpose(theta_1)))
    h = sigmoid(np.matmul(bias(layer_1), np.transpose(theta_2)))
    return np.argmax(h)


mndata = MNIST('samples')
mndata.gz = True
images, labels = mndata.load_testing()
theta_1 = np.genfromtxt('theta_1.csv', delimiter=",")
theta_2 = np.genfromtxt('theta_2.csv', delimiter=",")
index = random.randrange(0, len(images))
prediction = predict(theta_1, theta_2, np.asarray(images)[index])
print(mndata.display(images[index]))
print('This was labeled ', labels[index])
print('I think this is ', prediction)

