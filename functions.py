# Arquivo para definir as funções de ativação, erro, e demias funções
# utilizadas pela rede neural.

from numpy import tanh, exp

def tanh(x):
    return tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x)**2

def logistic(x):
    return 1 / (1 + exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))