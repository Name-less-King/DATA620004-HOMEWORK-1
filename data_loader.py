from torchvision import datasets
import numpy as np
from utils import *


def data_loader():
    train = datasets.MNIST('./data', train=True, download=True)
    test = datasets.MNIST('./data', train=False, download=True)

    train_arr = train.data.numpy()
    test_arr = test.data.numpy()
    train_targets_arr = train.targets.numpy()
    test_targets_arr = test.targets.numpy()

    # normalization
    x_train = ((train_arr / 255) - 0.1307) / 0.3081
    y_train = train_targets_arr
    x_test = ((test_arr / 255) - 0.1307) / 0.3081
    y_test = test_targets_arr

    # extract 5000 pictures from training set as validation set
    index = [i for i in range(x_train.shape[0])]
    np.random.shuffle(index)
    x_val = x_train[index[0:5000], :, :]
    y_val = y_train[index[0:5000]]
    x_train = x_train[index[5000:60000], :, :]
    y_train = y_train[index[5000:60000]]

    # get the shape of vectors for later process
    length_train = x_train.shape[0]
    length_val = x_val.shape[0]
    length_test = x_test.shape[0]
    reshape_dim = train_arr.shape[1] * train_arr.shape[1]

    # vectorize the set and get the one-hot label
    x_train_flatten = x_train.reshape(length_train, reshape_dim)
    x_val_flatten = x_val.reshape(length_val, reshape_dim)
    x_test_flatten = x_test.reshape(length_test, reshape_dim)
    y_train_onehot = OneHot(y_train)
    y_val_onehot = OneHot(y_val)
    y_test_onehot = OneHot(y_test)

    return [x_train_flatten, x_val_flatten, x_test_flatten, y_train_onehot, y_val_onehot, y_test_onehot]