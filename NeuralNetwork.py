import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
# Usando a função logística do scipy pois se eu tentar
# fazer na mão, dá overflow.
from scipy.special import expit

SEED = 666

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

def logistic(x):
  return expit(x)

def logistic_derivative(x):
  return logistic(x) * (1 - logistic(x))

# Prepara the targets for the neural network
# The logic is: if the target is 1, the output should be [1, 0, 0, 0, 0]
def prepare_targets(targets, act_func):
    n = targets.shape[0]
    # Doing this in a smarth way
    # First, we create a matrix with the same number of rows as the targets, but with 5 columns
    # Then, we fill the matrix with 0
    # Finally, we replace the column that corresponds to the target with 1
    # For example, if the target is 3, we replace the 3rd column with 1, etc.
    targets_matrix = np.zeros((n, 5))
    # The targets are 1-indexed, so we need to subtract 1 to get the correct column
    targets_matrix[np.arange(n), targets - 1] = 1
    # If the activation function is tanh, we need to change the 0s to -1s
    if act_func == 'tanh':
        targets_matrix[targets_matrix == 0] = -1
    return targets_matrix

np.random.seed(SEED)

class MultiClassClassificationNetwork:
    def __init__(self, data_path, act_func = 'tanh', n_hidden = None):
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

    def train(self, epochs):
        for _ in range(epochs):
            self.forward(self.inputs)
            self.__backward()

    def test(self, data_path):
        data = pd.read_csv(data_path)
        # A última coluna é o target
        targets = data.iloc[:, -1].values
        inputs = data.iloc[:, :-1].values
        predictions = self.forward(inputs)
        predictions = np.argmax(predictions, axis=1) + 1
        cm = confusion_matrix(targets, predictions)
        return cm

classifier = MultiClassClassificationNetwork('data/treinamento.csv')

classifier.train(1000)

cm = classifier.test('data/teste.csv')
print(cm)