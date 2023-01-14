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
    targets_matrix[targets_matrix == 0] = -1

    return targets_matrix.reshape(n, n_outputs).astype(int)

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

def logistic(x):
  return np.exp(-np.logaddexp(-x, 0)) * 2 - 1

def logistic_derivative(x):
  sigmoid = logistic(x)
  return 0.5 * (1 + sigmoid) * (1 - sigmoid)

act = logistic
act_derivative = logistic_derivative

class MultiClassClassificationNetwork:
  def __init__(self, input_size, output_size):
    #np.random.seed(42)
    #np.random.seed(11092001)
    np.random.seed(666)
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_layers = int(np.sqrt(input_size * output_size))
    #self.weights1 = np.random.rand(self.input_size, self.hidden_layers)
    #self.weights2 = np.random.rand(self.hidden_layers, self.output_size)
    self.weights1 = np.random.uniform(-0.01, 0.01, (self.input_size, self.hidden_layers))
    self.weights2 = np.random.uniform(-0.01, 0.01, (self.hidden_layers, self.output_size))

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

# normalize the inputs to be between 0 and 1
inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())

# tanh = 80
# logistic = 40
classifier.train(inputs, targets, 5)

# testing

test_data = pd.read_csv('data/teste.csv')
# the other columns are the inputs
test_inputs = test_data.iloc[:, :-1].values
#test_inputs = np.array([[i] for i in test_inputs])
# normalize the inputs to be between 0 and 1
test_inputs = (test_inputs - test_inputs.min()) / (test_inputs.max() - test_inputs.min())

classifier.forward(test_inputs)
print(classifier.output)
preds = np.argmax(classifier.output, axis=1) + 1
cm = confusion_matrix(test_data.iloc[:, -1].values, preds)
print(cm)