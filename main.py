from NeuralNetwork import NeuralNetwork
from data_utils import import_data
from MainWindow import MainWindow
from PySide6.QtWidgets import QApplication
import sys

'''inputs, targets, input_size, output_size, n_hidden = import_data('data/treinamento.csv')

nn = NeuralNetwork(inputs, targets, input_size, output_size, n_hidden)
nn.train(1000)
cm = nn.test('data/teste.csv')
print(cm)'''

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())