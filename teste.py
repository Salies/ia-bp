import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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
    return targets_matrix.astype(int)

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

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
    self.hidden_layer = tanh(np.dot(inputs, self.weights1))
    self.output = tanh(np.dot(self.hidden_layer, self.weights2))
    return self.output

  def backward(self, targets):
    self.errors = targets - self.output
    self.output_deltas = self.errors * tanh_derivative(self.output)
    self.errors_hidden = self.output_deltas.dot(self.weights2.T)
    self.hidden_deltas = self.errors_hidden * tanh_derivative(self.hidden_layer)
    self.weights2 += self.hidden_layer.T.dot(self.output_deltas)
    self.weights1 += self.inputs.T.dot(self.hidden_deltas)
    #print(self.errors)

  def train(self, inputs, targets, epochs):
    # For each input and each target
    for _ in range(epochs):
        for input, target in zip(inputs, targets):
            i = [input]
            t = [target]
            i = np.array(i)
            t = np.array(t)
            self.forward(i)
            self.backward(t)

classifier = MultiClassClassificationNetwork(6, 5)

data = pd.read_csv('data/treinamento.csv')

# the last column is the target
targets = data.iloc[:, -1].values

# the other columns are the inputs
inputs = data.iloc[:, :-1].values

targets = prepare_targets(targets, 'tanh', 5)

classifier.train(inputs, targets, 250)

# testing

test_data = pd.read_csv('data/teste.csv')
# the other columns are the inputs
test_inputs = test_data.iloc[:, :-1].values

classifier.forward(test_inputs)
preds = np.argmax(classifier.output, axis=1) + 1
cm = confusion_matrix(test_data.iloc[:, -1].values, preds)
print(cm)