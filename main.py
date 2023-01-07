from NeuralNetwork import NeuralNetwork
from data_utils import import_data

inputs, targets, input_size, output_size = import_data('data/treinamento.csv')

nn = NeuralNetwork(inputs, targets, input_size, output_size, 5)
nn.train(1000)
cm = nn.test('data/teste.csv')
print(cm)