import numpy as np
import pandas as pd
from utils import *

class NeuralNetwork:
    def __init__(self, inputs, targets, input_size, output_size, n_hidden, stop_criteria = 'itr', act_func = 'tanh'):
        np.random.seed(666)
        # Normaliza as entradas
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        self.inputs = inputs
        # Prepara os targets -- por exemplo, se o target é 3, o output deve ser [-1, -1, 1, -1, -1]
        self.targets = prepare_targets(targets, output_size)
        self.input_size = input_size
        self.output_size = output_size
        # Número de neurônios na camada oculta
        self.n_hidden_layers = n_hidden
        # Critério de parada: número de iterações ou erro
        self.stop_criteria = stop_criteria

        # Atribui pesos iniciais aleatórios
        self.hidden_layer_weights = np.random.uniform(-0.01, 0.01, (self.input_size, self.n_hidden_layers))
        self.output_layer_weights = np.random.uniform(-0.01, 0.01, (self.n_hidden_layers, self.output_size))

        if act_func == 'tanh':
            self.act_func = tanh
            self.act_func_derivative = tanh_derivative
        elif act_func == 'logistic':
            self.act_func = logistic
            self.act_func_derivative = logistic_derivative

    def forward(self, inputs):
        # Calcula os nets da camada oculta
        net_hidden = inputs @ self.hidden_layer_weights
        # Calcula as saídas da camada oculta
        self.hidden_layers = self.act_func(net_hidden)
        # Calcula os nets da camada de saída
        net_output = self.hidden_layers @ self.output_layer_weights
        # Calcula as saídas da camada de saída
        self.output = self.act_func(net_output)
        return self.output

    def __backward(self, inputs, targets):
        # Calcula os erros da camada de saída
        self.output_errors = targets - self.output
        # Calcula os deltas da camada de saída
        self.output_deltas = self.output_errors * self.act_func_derivative(self.output)
        # Calcula os erros da camada oculta
        self.errors_hidden = self.output_deltas @ self.output_layer_weights.T
        # Calcula os deltas da camada oculta
        self.hidden_deltas = self.errors_hidden * self.act_func_derivative(self.hidden_layers)
        # Atualiza os pesos
        self.output_layer_weights += np.vstack(self.hidden_layers) @ np.reshape(self.output_deltas, (-1, self.output_deltas.shape[0]))
        self.hidden_layer_weights += np.vstack(inputs) @ np.reshape(self.hidden_deltas, (-1, self.hidden_deltas.shape[0]))

    # Função de treinamento
    def train(self, stop_value):
        # Prepara a entrada
        train_data = list(zip(self.inputs, self.targets))
        train_data_size = len(train_data)
        # Por número de iterações
        if self.stop_criteria == 'itr':
            for itr in range(stop_value):
                input, target = train_data[itr % train_data_size]
                self.forward(input)
                self.__backward(input, target)
        # Por erro
        elif self.stop_criteria == 'error':
            error = error_function(self.targets, self.forward(self.inputs))
            itr = 0
            while error > stop_value:
                input, target = train_data[itr % train_data_size]
                self.forward(input)
                self.__backward(input, target)
                error = error_function(self.targets, self.forward(self.inputs))
                itr += 1

    # Função de teste
    def test(self, data_path):
        data = pd.read_csv(data_path)
        # A última coluna é o target
        targets = data.iloc[:, -1].values - 1
        inputs = data.iloc[:, :-1].values
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        predictions = self.forward(inputs)
        # Função argmax retorna o índice do maior valor, que é o target predito pela rede
        predictions = np.argmax(predictions, axis=1)
        cm = confusion_matrix(targets, predictions)
        return cm