import numpy as np
from matplotlib import pyplot as plt
from utils import *

# This is the two layers NNs model.
class NN():
    def __init__(self, sizes, epochs, activation="sigmoid", learning_rate=0.01, weight_decay=0.0001, print=True):
        '''
        Input:
            sizes: [input_size,hidden_size,output_size],
            epochs: number of iterations to train,
            activation: "sigmoid" or "relu",
            learning_rate: the learning rate,
            weigh_decay:  l2 regularization,
            print: print the process or not.
        '''
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation = activation
        self.weight_decay = weight_decay
        self.print = print

        input_layer = sizes[0]
        hidden_layer = sizes[1]
        output_layer = sizes[2]

        # Xiaver initialization
        self.params = {
            'w1': np.random.randn(input_layer, hidden_layer) / np.sqrt(hidden_layer),
            'w2': np.random.randn(hidden_layer, output_layer) / np.sqrt(output_layer)
        }


    def forward(self, x):
        # forward computing
        activation = self.activation
        if activation == "sigmoid":
            activation = sigmoid
        elif activation == "relu":
            activation = relu
        else:
            raise Exception('Non-supported activation function')
        params = self.params
        memory = {}
        memory['a0'] = x

        memory['z1'] = memory['a0'] @ params["w1"]
        memory['a1'] = activation(memory['z1'])

        memory['z2'] = memory['a1'] @ params["w2"]
        memory['a2'] = softmax(memory['z2'])
        self.memory = memory
        return memory['a2']


    def loss(self, output, y_onehot):
        # cross-entropy loss function and weight decay
        params = self.params
        weight_decay = self.weight_decay
        n = y_onehot.shape[0]
        y = np.argmax(y_onehot, axis=1)
        loss = 0
        for key, value in params.items():
            loss = loss + weight_decay * np.sum(params[key] ** 2)
        for i in range(n):
            loss = loss - np.log(output[i, y[i]])
        return loss

    
    def backward(self, output, y_train):
        # backward propagation
        activation = self.activation
        if activation == "sigmoid":
            activation_backward = sigmoid_backward
        elif activation == "relu":
            activation_backward = relu_backward
        else:
            raise Exception('Non-supported activation function')
        params = self.params
        memory = self.memory
        dw = {}

        # compute the gradient of w2
        backward = (output - y_train)
        dw['w2'] = np.outer(memory['a1'], backward) + 2 * self.weight_decay * params['w2']

        # compute the gradient of w1
        backward = np.dot(params['w2'], backward) * activation_backward(memory['z1'])
        dw['w1'] = np.outer(memory['a0'], backward) + 2 * self.weight_decay * params['w1']
        return dw


    def learning_rate_decay(self, step):
        return self.learning_rate * 0.9 ** (step / 10000)    


    def predict(self, x):
        output = self.forward(x)
        y_hat = np.argmax(output, axis=1)
        return y_hat


    def accuracy(self, x_test, y_test):
        y = np.argmax(y_test, axis=1)
        n = x_test.shape[0]
        correct = 0
        for index, x in enumerate(x_test):
            output = self.forward(x)
            pred = np.argmax(output)
            if pred == y[index]:
                correct = correct + 1
        return correct / n


    def train(self, x_train, y_train_onehot, x_val, y_val_onehot, plot=False):
        n_train = x_train.shape[0]
        epochs = self.epochs
        if plot == True:
            # If plot == True, store the loss
            loss_train_arr = np.zeros(51)
            loss_val_arr = np.zeros(51)
            acc_train_arr = np.zeros(51)
            acc_val_arr = np.zeros(51)
            num = 0

            for iteration in range(epochs):
                rand_i = int(np.random.rand(1) * n_train)
                x_rand = x_train[rand_i, :]

                output = self.forward(x_rand)
                dw = self.backward(output, y_train_onehot[rand_i, :])

                for key, value in dw.items():
                    self.params[key] -= self.learning_rate_decay(iteration) * value
                
                if (iteration % (epochs // 10) == 0) and (self.output == True):
                    accuracy = self.accuracy(x_val, y_val_onehot)
                    print("Epoch: {0}, Accuracy in the test set: {1:.2f}%".format(iteration+1, accuracy * 100))
                
                if ((iteration % (epochs // 50) == 0)):
                    out_train = self.forward(x_train)
                    out_val = self.forward(x_val)
                    acc_train_arr[num] = self.accuracy(x_train, y_train_onehot)
                    acc_val_arr[num] = self.accuracy(x_val, y_val_onehot)
                    loss_train_arr[num] = self.loss_without_reg(out_train, y_train_onehot)
                    loss_val_arr[num] = self.loss_without_reg(out_val, y_val_onehot)
                    num = num + 1

            out_train = self.forward(x_train)
            out_val = self.forward(x_val)
            acc_train_arr[num] = self.accuracy(x_train, y_train_onehot)
            acc_val_arr[num] = self.accuracy(x_val, y_val_onehot)
            loss_train_arr[num] = self.loss_without_reg(out_train, y_train_onehot)
            loss_val_arr[num] = self.loss_without_reg(out_val, y_val_onehot)
            
            self.acc_train_arr = acc_train_arr
            self.acc_val_arr = acc_val_arr
            self.loss_train_arr = loss_train_arr
            self.loss_val_arr = loss_val_arr
        else:
            for iteration in range(epochs):
                rand_i = int(np.random.rand(1) * n_train)
                x_rand = x_train[rand_i, :]

                output = self.forward(x_rand)
                dw = self.backward(output, y_train_onehot[rand_i, :])

                for key, value in dw.items():
                    self.params[key] -= self.learning_rate_decay(iteration) * value
                
                if (iteration % (epochs // 10) == 0) and (self.print == True):
                    accuracy = self.accuracy(x_val, y_val_onehot)
                    print("Epoch: {0}, Accuracy in the validation set: {1:.2f}%".format(iteration+1, accuracy * 100))
                    
        accuracy = self.accuracy(x_val, y_val_onehot)
        print("Epoch: {0}, Final accuracy in the validation set: {1:.2f}%".format(iteration+1, accuracy * 100))
        return accuracy


    def show_accuracy_plot(self):
        epochs = self.epochs
        acc_train = self.acc_train_arr
        acc_val = self.acc_val_arr
        i = np.arange(0, 51, 1) * (epochs / 50)
        plt.plot(i, acc_train, i, acc_val)
        plt.grid()
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(["Training set", "Test set"])

    def show_loss_plot(self):
        epochs = self.epochs
        loss_train = self.loss_train_arr
        loss_val = self.loss_val_arr
        i = np.arange(0, 51, 1) * (epochs / 50)
        plt.plot(i, loss_train, i, loss_val)
        plt.grid()
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Training set", "Test set"])