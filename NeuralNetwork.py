import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils import *

class NeuralNetwork:
    def __init__(self, inputs, targets, input_size, output_size, n_hidden, stop_criteria = 'epochs', act_func = 'tanh'):
        np.random.seed(666)
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        self.inputs = inputs
        self.targets = prepare_targets(targets, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden
        self.stop_criteria = stop_criteria

        # self.weights1 = np.random.rand(self.input_size, self.n_hidden_layers)
        # self.weights2 = np.random.rand(self.n_hidden_layers, self.output_size)
        self.weights1 = np.random.uniform(-0.01, 0.01, (self.input_size, self.n_hidden_layers))
        self.weights2 = np.random.uniform(-0.01, 0.01, (self.n_hidden_layers, self.output_size))

        if act_func == 'tanh':
            self.act_func = tanh
            self.act_func_derivative = tanh_derivative
        elif act_func == 'logistic':
            self.act_func = logistic
            self.act_func_derivative = logistic_derivative

    def forward(self, inputs):
        net_hidden = inputs @ self.weights1
        self.hidden_layers = self.act_func(net_hidden)
        net_output = self.hidden_layers @ self.weights2
        self.output = self.act_func(net_output)
        return self.output

    def __backward(self, inputs, targets):
        self.errors = targets - self.output
        self.output_deltas = self.errors * self.act_func_derivative(self.output)
        self.errors_hidden = self.output_deltas @ self.weights2.T
        self.hidden_deltas = self.errors_hidden * self.act_func_derivative(self.hidden_layers)
        self.weights2 += np.vstack(self.hidden_layers) @ np.reshape(self.output_deltas, (-1, self.output_deltas.shape[0]))
        self.weights1 += np.vstack(inputs) @ np.reshape(self.hidden_deltas, (-1, self.hidden_deltas.shape[0]))

    def train(self, stop_value):
        train_data = list(zip(self.inputs, self.targets))
        train_data_size = len(train_data)
        if self.stop_criteria == 'epochs':
            for itr in range(stop_value):
                input, target = train_data[itr % train_data_size]
                self.forward(input)
                self.__backward(input, target)
            # for _ in range(stop_value):
            #     for input, target in zip(self.inputs, self.targets):
            #         self.forward(input)
            #         self.__backward(input, target)
        elif self.stop_criteria == 'error':
            error = error_function(self.targets, self.forward(self.inputs))
            itr = 0
            while error > stop_value:
                input, target = train_data[itr % train_data_size]
                self.forward(input)
                self.__backward(input, target)
                error = error_function(self.targets, self.forward(self.inputs))
                itr += 1

    def test(self, data_path):
        data = pd.read_csv(data_path)
        # A última coluna é o target
        targets = data.iloc[:, -1].values
        inputs = data.iloc[:, :-1].values
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        predictions = self.forward(inputs)
        predictions = np.argmax(predictions, axis=1) + 1
        cm = confusion_matrix(targets, predictions)
        return cm