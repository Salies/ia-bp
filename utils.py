# Arquivo para definir as funções de ativação, erro, e demias funções utilizadas pela rede neural.

import numpy as np
# Usando a função logística do scipy pois se eu tentar
# fazer na mão, dá overflow.
from scipy.special import expit

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

def logistic(x):
  return expit(x)

def logistic_derivative(x):
  return logistic(x) * (1 - logistic(x))

# Error function is half squared error
def error_function(targets, outputs):
    return np.sum((targets - outputs)**2) / 2