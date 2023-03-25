# utils for training a neural network
import numpy as np

# ReLu function
def relu(x):
    return np.maximum(0, x)

def relu_backward(x):
    dx = np.zeros(x.shape)
    dx[x > 0] = 1
    return dx
     
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x):
    dx = np.exp(-x) / (np.exp(-x)+1)**2
    return dx

# Softmax
def softmax(x):
    # add a small number 1e-6 to prevent overflow 
    exps = np.exp(x) + 1e-6
    if (len(x.shape) == 1):
        return exps / np.sum(exps, axis=0)
    else:
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

def softmax_backward(x):
    exps = np.exp(x) + 1e-6
    if (len(x.shape) == 1):
        dx = exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return dx
    else:
        dx = exps / np.sum(exps, axis=1).reshape(-1, 1) * (1 - exps / np.sum(exps, axis=0).reshape(-1, 1))
        return dx

# get an one-hot vector
def OneHot(x):
    labels = np.max(x) + 1
    one_hot = np.zeros((x.shape[0], labels))
    for index, i in enumerate(x):
        one_hot[index, i] = 1
    return one_hot