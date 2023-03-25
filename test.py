from torchvision import datasets
import numpy as np
from torch import load
from matplotlib import pyplot as plt
from data_loader import *
from nets import *

# plot or not
show_plot = False

# load the MNIST dataset
data = data_loader()
x_train_flatten = data[0]
x_val_flatten = data[1]
x_test_flatten = data[2]
y_train_onehot = data[3]
y_val_onehot = data[4]
y_test_onehot = data[5]

# load the best model
parameters = load("./params")

w = parameters[0]
activation = parameters[1]
learning_rate = parameters[2]
hidden_size = parameters[3]
weight_decay = parameters[4]
epochs = 200000

nn_best = NN([784, hidden_size, 10], epochs, activation=activation, learning_rate=learning_rate, weight_decay=weight_decay)
nn_best.params = w
y_hat = nn_best.predict(x_test_flatten)
accuracy = nn_best.accuracy(x_test_flatten,y_test_onehot)
print("The accuracy of the model in the test set is: {0:.2f}%.".format(accuracy * 100))

# plot the results
if show_plot == True:
    print("Retrain and show the accuracy and loss plot.")
    nn_best_retrain = NN([784, hidden_size, 10], epochs, activation=activation, learning_rate=learning_rate, weight_decay=weight_decay)
    acc = nn_best_retrain.train(x_train_flatten, y_train_onehot, x_test_flatten, y_test_onehot, plot=True)

    nn_best_retrain.show_accuracy_plot()
    plt.savefig("./accuracy_plot")
    plt.pause(6)
    plt.close()
    nn_best_retrain.show_loss_plot()
    plt.savefig("./loss_plot")
    plt.pause(6)
