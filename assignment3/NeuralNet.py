#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

class NeuralNet:

    def __init__(self, train, header = True, h1 = 20, h2 = 10, activation = 'sigmoid'):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers
        self.activation = activation
        if header == False:
            raw_input = pd.read_csv(train, na_values=['?', ' ?'], header=None)
        else:
            raw_input = pd.read_csv(train, na_values=['?', ' ?'])
        # TODO: Remember to implement the preprocess method
        dataset = self.preprocess(raw_input)
        ncols = len(dataset.columns)
        nrows = len(dataset.index)
        X = dataset.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        y = dataset.iloc[:, (ncols - 1)].values.reshape(nrows, 1)
        enc = OneHotEncoder()
        y = enc.fit_transform(y).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        self.X = X_train
        self.y = y_train
        self.X_t = X_test
        self.y_t = y_test
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(self, x)
        if activation == "ReLu":
            self.__ReLu(self, x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)

        if activation == 'tanh':
            self.__tanh_derivative(self, x)

        if activation == 'ReLu':
            self.__ReLu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1. - x)

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __tanh_derivative(self, x):
        return 1. - x * x

    def __ReLu(self, x):
        return x * (x > 0)

    def __ReLu_derivative(self, x):
        return 1. * (x > 0)


    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):

        # label encode
        X[X.select_dtypes(['object']).columns] = X.select_dtypes(['object']).apply(lambda x: x.astype('category'))
        X[X.select_dtypes(['category']).columns] = X.select_dtypes(['category']).apply(lambda x: x.cat.codes)
        # fill the missing value
        X = X.apply(lambda x: x.fillna(x.value_counts().index[0]))
        #X = X.fillna(X.mean())
        # Standardize data
        scaler = StandardScaler()
        X.iloc[:, :-1] = scaler.fit_transform(X.iloc[:, :-1])
        print(X)
        return X

    # Below is the training function

    def train(self, max_iterations = 1000, learning_rate = 0.001):
        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, self.activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input
            print(str(np.sum(error)/len(self.X)))

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error/len(self.X))))
        # print("The final weight vectors are (starting from input to output layers)")
        # print(self.w01)
        # print(self.w12)
        # print(self.w23)

    def forward_pass(self):
        # pass our inputs through our neural network
        if self.activation == 'sigmoid':
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        if self.activation == 'tanh':
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        if self.activation == 'ReLu':
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__ReLu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__ReLu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__ReLu(in3)
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if activation == "ReLu":
            delta_output = (self.y - out) * (self.__ReLu_derivative(out))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        if activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__ReLu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        if activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__ReLu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        if activation == "ReLu":
            delta_input_layer = np.multiply(self.__ReLu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test=None, header = True):
        if test == None:
            self.X = self.X_t
            self.y = self.y_t
        else:
            if header == False:
                raw_input = pd.read_csv(test, na_values='?', header=None)
            else:
                raw_input = pd.read_csv(test, na_values='?')
            test_dataset = self.preprocess(raw_input)
            ncols = len(test_dataset.columns)
            nrows = len(test_dataset.index)
            self.X = test_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
            self.y = test_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
            enc = OneHotEncoder()
            self.y = enc.fit_transform(self.y).toarray()
        out = self.forward_pass()
        out = (out == out.max(axis=1, keepdims=1)).astype(int)
        return accuracy_score(self.y, out)

if __name__ == "__main__":
    url_iris = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    url_car = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    url_adult = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    #url = 'train.csv'
    print('Welcome to ML6375 Neural Network!')
    flag = True
    while flag:
        data = int(input('Which data do you want to use?\n[1] iris.data [2] car.data [3] adult.data [4] I want to my own data'))
        if data == 1:
            url = url_iris
            flag = False
        elif data == 2:
            url = url_car
            flag = False
        elif data == 3:
            url = url_adult
            flag = False
        elif data == 4:
            url = str(input('Please write or paste the path or url'))
            flag = False
    flag = True
    while flag:
        data = int(input('Which activation do you want to use?\n[1] sigmoid [2] tanh [3] ReLu'))
        if data == 1:
            activation = 'sigmoid'
            flag = False
        elif data == 2:
            activation = 'tanh'
            flag = False
        elif data == 3:
            activation = 'ReLu'
            flag = False
    h1 = int(input('Please input first hidden layer size'))
    h2 = int(input('Please input second hidden layer size'))
    learing_rate = float(input('Please input learing_rate'))
    iteration = int(input('Please input max_iteration'))

    neural_network = NeuralNet(url, False, activation=activation, h1=h1, h2=h2)
    neural_network.train(learning_rate=learing_rate, max_iterations=iteration)
    testError = neural_network.predict()
    print("accuracy")
    print(testError)


