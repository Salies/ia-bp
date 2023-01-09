import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils import *

class NeuralNetwork:
    def __init__(self, inputs, targets, input_size, output_size, n_hidden, stop_criteria = 'epochs', act_func = 'tanh'):
        np.random.seed(666)
        self.inputs = inputs
        self.targets = prepare_targets(targets, act_func, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden
        self.stop_criteria = stop_criteria

        self.weights1 = np.random.rand(self.input_size, self.n_hidden_layers)
        self.weights2 = np.random.rand(self.n_hidden_layers, self.output_size)

        if act_func == 'tanh':
            self.act_func = tanh
            self.act_func_derivative = tanh_derivative
        elif act_func == 'logistic':
            self.act_func = logistic
            self.act_func_derivative = logistic_derivative

    def forward(self, inputs):
        self.hidden_layers = self.act_func(np.dot(inputs, self.weights1))
        self.output = self.act_func(np.dot(self.hidden_layers, self.weights2))
        return self.output

    def __backward(self):
        self.errors = self.targets - self.output
        self.output_deltas = self.errors * self.act_func_derivative(self.output)
        self.errors_hidden = self.output_deltas.dot(self.weights2.T)
        self.hidden_deltas = self.errors_hidden * self.act_func_derivative(self.hidden_layers)
        self.weights2 += self.hidden_layers.T.dot(self.output_deltas)
        self.weights1 += self.inputs.T.dot(self.hidden_deltas)

    def train(self, stop_value):
        if self.stop_criteria == 'epochs':
            for _ in range(stop_value):
                self.forward(self.inputs)
                self.__backward()
        elif self.stop_criteria == 'error':
            error = error_function(self.targets, self.forward(self.inputs))
            while error > stop_value:
                self.forward(self.inputs)
                self.__backward()
                error = error_function(self.targets, self.output)

    def test(self, data_path):
        data = pd.read_csv(data_path)
        # A última coluna é o target
        targets = data.iloc[:, -1].values
        inputs = data.iloc[:, :-1].values
        predictions = self.forward(inputs)
        predictions = np.argmax(predictions, axis=1) + 1
        cm = confusion_matrix(targets, predictions)
        return cm

    # Outras funções requeridas pelo enunciado.
    # "O programa deve indicar o número de neurônios na camada de entrada (6) e saída (5)"
    def get_neurons_sizes(self):
        return self.input_size, self.output_size