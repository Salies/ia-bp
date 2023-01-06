import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

def logistic(x):
  return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
  return logistic(x) * (1 - logistic(x))

act_func_s = 'logistic'

if act_func_s == 'tanh':
  act_func = tanh
  act_func_derivative = tanh_derivative
elif act_func_s == 'logistic':
  act_func = logistic
  act_func_derivative = logistic_derivative

np.random.seed(666)

class MultiClassClassificationNetwork:
  def __init__(self, input_size, output_size):
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_layers = int(np.sqrt(input_size*output_size))
    self.weights1 = np.random.rand(self.input_size, self.hidden_layers)
    self.weights2 = np.random.rand(self.hidden_layers, self.output_size)

  def forward(self, inputs):
    self.inputs = inputs
    self.hidden_layer = act_func(np.dot(inputs, self.weights1))
    self.output = act_func(np.dot(self.hidden_layer, self.weights2))
    return self.output

  def backward(self, targets):
    self.errors = targets - self.output
    self.output_deltas = self.errors * act_func_derivative(self.output)
    self.errors_hidden = self.output_deltas.dot(self.weights2.T)
    self.hidden_deltas = self.errors_hidden * act_func_derivative(self.hidden_layer)
    self.weights2 += self.hidden_layer.T.dot(self.output_deltas)
    self.weights1 += self.inputs.T.dot(self.hidden_deltas)

  def train(self, inputs, targets, epochs):
    for _ in range(epochs):
      self.forward(inputs)
      self.backward(targets)

classifier = MultiClassClassificationNetwork(6, 5)

# Loading training data
data = pd.read_csv('treinamento.csv')
# Shuffling the data
data = data.sample(frac=1, random_state=666).reset_index(drop=True)
#print(data)
# The last column is the target
X = data.iloc[:, :-1].values
y_data = data.iloc[:, -1].values
y = []
# The logic is: if the target is 1, the output should be [1, -1, -1, -1, -1]
# if the target is 2, the output should be [-1, 1, -1, -1, -1], etc.
if act_func_s == 'tanh':
  for i in range(len(y_data)):
    if y_data[i] == 1:
        y.append([1, -1, -1, -1, -1])
    elif y_data[i] == 2:
        y.append([-1, 1, -1, -1, -1])
    elif y_data[i] == 3:
        y.append([-1, -1, 1, -1, -1])
    elif y_data[i] == 4:
        y.append([-1, -1, -1, 1, -1])
    elif y_data[i] == 5:
        y.append([-1, -1, -1, -1, 1])
elif act_func_s == 'logistic':
  for i in range(len(y_data)):
    if y_data[i] == 1:
        y.append([1, 0, 0, 0, 0])
    elif y_data[i] == 2:
        y.append([0, 1, 0, 0, 0])
    elif y_data[i] == 3:
        y.append([0, 0, 1, 0, 0])
    elif y_data[i] == 4:
        y.append([0, 0, 0, 1, 0])
    elif y_data[i] == 5:
        y.append([0, 0, 0, 0, 1])

y = np.array(y)

# Training the model
classifier.train(X, y, 1000)

# Loading test data
data = pd.read_csv('teste.csv')
# The last column is the target
X_test = data.iloc[:, :-1].values
y_test = data.iloc[:, -1].values

# Predicting the test set results
y_pred = classifier.forward(X_test)

pred = np.argmax(y_pred, axis=1) + 1

# Making the Confusion Matrix
cm = confusion_matrix(y_test, pred)

print(cm)