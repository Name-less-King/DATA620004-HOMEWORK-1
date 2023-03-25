import numpy as np
from torch import save
from utils import *
from data_loader import *
from nets import *

# search hyperparameters
learning_rate = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
weight_decay = [1e-6, 1e-5, 1e-4]
hidden_size = [200, 225, 250, 275, 300]
activation = ["relu", "sigmoid"]
epochs = 200000

# load the MNIST dataset
data = data_loader()
x_train_flatten = data[0]
x_val_flatten = data[1]
x_test_flatten = data[2]
y_train_onehot = data[3]
y_val_onehot = data[4]
y_test_onehot = data[5]

# Store the best accuracy and hyperparameters
accuracy_best = 0
act_best, lr_best, wd_best, hs_best = None, None, None, None

# Training process
params = {}
count = 0

for act in activation:
    for lr in learning_rate:
        for wd in weight_decay:
            for hs in hidden_size:
                count = count + 1
                print("\nModel {0}".format(count))
                print("Activation function: {0}, step size: {1}, hidden size: {2}, lambda: {3}".format(act, lr, hs, wd))
                nn = NN([784, hs, 10], epochs, activation=act, learning_rate=lr, weight_decay=wd, print=False)
                acc = nn.train(x_train_flatten, y_train_onehot, x_val_flatten, y_val_onehot)
                if acc > accuracy_best:
                    accuracy_best = acc
                    params = nn.params
                    act_best = act
                    lr_best = lr
                    wd_best = wd
                    hs_best = hs


# print the best accuracy in the VALIDATION set and its hyperparameters.
print("Best accuracy in the validation set: {0:.2f}%".format(accuracy_best * 100))
print("Best model setting: activation function: {0}, learning rate: {1}, hidden size: {2}, weight decay: {3}".format(act_best, lr_best, hs_best, wd_best))

# save the model
save([params, act_best, lr_best, hs_best, wd_best], './params')
