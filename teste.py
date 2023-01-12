import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.special import expit as logistic_a

def prepare_targets(targets, act_func, n_outputs):
    n = targets.shape[0]
    # Doing this in a smarth way
    # First, we create a matrix with the same number of rows as the targets, but with 5 columns
    # Then, we fill the matrix with 0
    # Finally, we replace the column that corresponds to the target with 1
    # For example, if the target is 3, we replace the 3rd column with 1, etc.
    targets_matrix = np.zeros((n, n_outputs))
    # The targets are 1-indexed, so we need to subtract 1 to get the correct column
    targets_matrix[np.arange(n), targets - 1] = 1
    # If the activation function is tanh, we need to change the 0s to -1s
    if act_func == 'tanh':
        targets_matrix[targets_matrix == 0] = -1

    return targets_matrix.reshape(n, n_outputs).astype(int)

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

def logistic(x):
  return logistic_a(x) - 0.5

def logistic_derivative(x):
  return (logistic_a(x) * (1 - logistic_a(x))) - 0.5

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def softmax_derivative(x):
  return softmax(x) * (1 - softmax(x))

act = softmax
act_derivative = softmax_derivative

class MultiClassClassificationNetwork:
  def __init__(self, input_size, output_size):
    np.random.seed(666)
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_layers = int(np.sqrt(input_size * output_size))
    self.weights1 = np.random.rand(self.input_size, self.hidden_layers)
    self.weights2 = np.random.rand(self.hidden_layers, self.output_size)

  def forward(self, inputs):
    self.inputs = inputs
    net_hidden = inputs @ self.weights1
    self.hidden_layer = act(net_hidden)
    net_output = self.hidden_layer @ self.weights2
    self.output = act(net_output)
    return self.output

  def backward(self, targets):
    self.errors = targets - self.output
    self.output_deltas = self.errors * act_derivative(self.output)
    self.errors_hidden = self.output_deltas @ self.weights2.T
    self.hidden_deltas = self.errors_hidden * act_derivative(self.hidden_layer)
    self.weights2 += self.hidden_layer.T @ self.output_deltas
    self.weights1 += self.inputs.T @ self.hidden_deltas

  def train(self, inputs, targets, epochs):
    # For each input and each target
    data = list(zip(inputs, targets))
    for _ in range(epochs):
        for input, target in data:
            self.forward(input)
            self.backward(target)

classifier = MultiClassClassificationNetwork(6, 5)

data = pd.read_csv('data/treinamento.csv')
data = data.sample(frac=1, random_state=666).reset_index(drop=True)

# the last column is the target
targets = data.iloc[:, -1].values

# the other columns are the inputs
inputs = data.iloc[:, :-1].values

targets = prepare_targets(targets, act.__name__, 5)

inputs = np.array([[i] for i in inputs])

classifier.train(inputs, targets, 400)

# testing

test_data = pd.read_csv('data/teste.csv')
# the other columns are the inputs
test_inputs = test_data.iloc[:, :-1].values

classifier.forward(test_inputs)
preds = np.argmax(classifier.output, axis=1) + 1
cm = confusion_matrix(test_data.iloc[:, -1].values, preds)
print(cm)