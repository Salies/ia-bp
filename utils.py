# Arquivo para definir as funções de ativação, erro, e demias funções utilizadas pela rede neural.

import numpy as np

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

def logistic(x):
  return np.exp(-np.logaddexp(-x, 0)) * 2 - 1

def logistic_derivative(x):
  return 0.5 * (1 + logistic(x)) * (1 - logistic(x))

# Error function is half squared error
def error_function(targets, outputs):
    return np.sum((targets - outputs)**2) / 2

# Prepara the targets for the neural network
# The logic is: if the target is 1, the output should be [1, 0, 0, 0, 0]
def prepare_targets(targets, n_outputs):
    n = targets.shape[0]
    # Doing this in a smarth way
    # First, we create a matrix with the same number of rows as the targets, but with 5 columns
    # Then, we fill the matrix with 0
    # Finally, we replace the column that corresponds to the target with 1
    # For example, if the target is 3, we replace the 3rd column with 1, etc.
    targets_matrix = np.full((n, n_outputs), fill_value=-1)
    # The targets are 1-indexed, so we need to subtract 1 to get the correct column
    targets_matrix[np.arange(n), targets - 1] = 1
    return targets_matrix