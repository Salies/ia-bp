import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Open the training data file
data = pd.read_csv('treinamento.csv')

#print(data)

# Open the test data file
test = pd.read_csv('teste.csv')

#print(test)

# Normalize the data to be between 0 and 1
data = (data - data.min()) / (data.max() - data.min())
test = (test - test.min()) / (test.max() - test.min())

X_train = data.iloc[:, 0:6]
y_train = data.iloc[:, 6].astype(int)

X_test = test.iloc[:, 0:6]
y_test = test.iloc[:, 6].astype(int)

# Initializing hyperparameters
learning_rate = 0.1
epochs = 100000
N = len(X_train)

# number of input features
input_size = 6

# number of neurons in the output layer
output_size = 5

# number of neurons in the hidden layer is the geometric mean of input and output
hidden_size = np.round(np.sqrt(input_size * output_size)).astype(int)

# Initializing weights
np.random.seed(42)

# initializing weight for the hidden layer
W1 = np.random.normal(scale=0.1, size=(input_size, hidden_size))   

# initializing weight for the output layer
W2 = np.random.normal(scale=0.1, size=(hidden_size, output_size))

# Helper functions
def logistic(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

'''def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()'''

y_train = y_train[:, np.newaxis]
for itr in range(epochs):    
    
    # feedforward propagation
    # on hidden layer
    Z1 = np.dot(X_train, W1)
    A1 = logistic(Z1)

    # on output layer
    Z2 = np.dot(A1, W2)
    A2 = logistic(Z2)
    
    # backpropagation
    E1 = A2 - y_train
    dW1 = E1 * A2 * (1 - A2)

    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)
    
    # weight updates
    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(X_train.T, dW2) / N

    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update

    loss = mean_squared_error(y_train, A2)

    if loss < 0.01:
        print("Acabou o treinamento")
        break

Z1 = np.dot(X_test, W1)
A1 = logistic(Z1)

Z2 = np.dot(A1, W2)
A2 = logistic(Z2)

y_pred = A2.argmax(axis=1)
y_true = y_test

acc = y_pred == y_true
print(acc.mean())