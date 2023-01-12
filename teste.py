import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

class MultiClassClassificationNetwork:
  def __init__(self, input_size, output_size, hidden_size):
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_layers = hidden_size
    #self.weights1 = np.random.rand(self.input_size, self.hidden_layers)
    #self.weights2 = np.random.rand(self.hidden_layers, self.output_size)
    self.weights2 = np.vstack([
        1.2, 1.6, 4.3, 3.2
    ])
    w1 = [
        [1.1, 3.6, 2.1, 0.9],
        [-1.4, -4.1, 2.5, -1.0]
    ]
    self.weights1 = np.array(w1)

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
    print(self.errors)

  def train(self, inputs, targets, epochs):
    for _ in range(epochs):
      self.forward(inputs)
      self.backward(targets)

classifier = MultiClassClassificationNetwork(2, 1, 4)

inputs = [[0, 1]]
targets = [[1]]

inputs = np.array(inputs)
targets = np.array(targets)

classifier.forward(inputs)
classifier.backward(targets)
