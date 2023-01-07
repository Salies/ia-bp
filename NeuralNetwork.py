import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils import *

SEED = 666

np.random.seed(SEED)

class NeuralNetwork:
    def __init__(self, data_path, stop_criteria = 'epochs', act_func = 'tanh', n_hidden = None):
        data = pd.read_csv(data_path)
        # Embaralha os dados para que o treinamento seja mais eficiente.
        data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)
        # A última coluna é o target
        self.targets = data.iloc[:, -1].values.astype(int)
        self.output_size = len(np.unique(self.targets))
        self.targets = prepare_targets(self.targets, act_func)
        self.inputs = data.iloc[:, :-1].values
        self.input_size = self.inputs.shape[1]
        self.n_hidden_layers = n_hidden
        if self.n_hidden_layers is None:
            self.n_hidden_layers = int(np.sqrt(self.input_size*self.output_size))
        self.weights1 = np.random.rand(self.input_size, self.n_hidden_layers)
        self.weights2 = np.random.rand(self.n_hidden_layers, self.output_size)

        self.stop_criteria = stop_criteria

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