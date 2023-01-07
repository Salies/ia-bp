from functions import *
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Uma seed para as coisas aleatórias deste trabalho,
# assim os testes serão consistentes. Lembrando que uma seed fixa
# não faz com que os resultados deixem de ser aleatórios.
SEED = 666

# Classe para a rede neural do trabalho.
# Mais especificamente, trata-se de uma rede neural classificadora
# multiclasse e multicamada.

class NeuralNetwork:
    # Construtor da classe. A função de ativação pode ser passada para ela.
    # Por padrão, é a tangente hiperbólica.
    def __init__(self, a_fun='tanh'):
        self.set_a_fun(a_fun)

    # Função para preparar os targets de acordo com a função de ativação.
    # Por exemplo, se a classe for 1 e função logística, 
    # o target será [1, 0, 0, 0, 0]. Se fosse tanh, seria [1, -1, -1, -1, -1].
    def __prepare_targets(self, targets):
        # Fazendo isso de um jeito "esperto".
        # Primeiro criamos uma matriz de zeros com o número de colunas igual
        # ao número de saídas e o número de linhas igual ao tamanho do dataset.
        targets_matrix = np.zeros((len(targets), self.output_size)).astype(int)
        # Agora, para cada linha da matriz, setamos o valor da coluna [classe - 1] como 1.
        targets_matrix[np.arange(len(targets)), targets - 1] = 1
        # Se a função de ativação for tanh, setamos tudo o que for 0 como -1.
        if self.a_fun == tanh:
            targets_matrix[targets_matrix == 0] = -1
        # Caso contrário, basta retornar.
        return targets_matrix

    # Método para entrar com os dados de treinamento.
    # O método detecta o número de neurônios de entrada e de saída,
    # e estima o número de neurônios da camada oculta, mas este
    # pode ser alterado manualmente, como manda o enunciado.
    def set_data(self, path):
        data = pd.read_csv(path)
        # As classes são a última coluna do dataset.
        # Chama-se as classes de targets utilizando linguajar comum
        # de redes neurais. Ex.: https://www.deeplearningbook.org/contents/mlp.html
        targets = data.iloc[:, -1]
        # As entradas são todas as colunas exceto a última.
        self.inputs = data.iloc[:, :-1]
        # Salva os tamanhos
        self.input_size = self.inputs.shape[1]
        self.output_size = len(targets.unique())
        # Prepara os targets de acordo com a função de ativação.
        self.targets = self.__prepare_targets(targets)
        # Estima o número de neurônios da camada oculta
        self.n_hidden_layers = int(np.sqrt(self.input_size*self.output_size))
        # Inicializa os pesos.
        self.__init_weights()

    # Função para inicializar os pesos aleatoriamente.
    def __init_weights(self):
        self.output_layer_weights = np.random.rand(self.input_size, self.hidden_layers)
        self.hidden_layer_weights = np.random.rand(self.hidden_layers, self.output_size)

    # Função para alterar o número de neurônios da camada oculta, caso
    # o usuário deseje.
    def set_hidden_layers(self, n):
        self.hidden_layers = n
        # Re-inicializa os pesos.
        self.__init_weights()

    # Funções de treinamento.
    # Propara os resultados da função de ativação para frente.
    def __forward(self):
        self.hidden_layers = self.a_fun(np.dot(self.inputs, self.output_layer_weights))
        self.outputs = self.a_fun(np.dot(self.hidden_layers, self.hidden_layer_weights))

    # Backpropagation.
    def __backward(self):
        errors = self.targets - self.output
        output_deltas = errors * self.a_fun_d(self.output)
        errors_hidden = output_deltas.dot(self.hidden_layer_weights.T)
        hidden_deltas = errors_hidden * self.a_fun_d(self.hidden_layers)
        self.hidden_layer_weights += self.hidden_layers.T.dot(output_deltas)
        self.output_layer_weights += self.inputs.T.dot(hidden_deltas)

    # Função para calcular o erro da rede.
    def __error(self):
        return np.sum(np.square(self.targets - self.output))/2

    def train(self, stop_criteria = 'epochs', epochs=1000, min_error=0.01):
        if stop_criteria == 'epochs':
            for _ in range(epochs):
                self.__forward()
                self.__backward()
            return
        
        if stop_criteria == 'error':
            error = self.__error()
            while error > min_error:
                self.__forward()
                self.__backward()
                error = self.__error()
            return

    # Função de teste, dado um dataset de teste.
    def test(self, path):
        data = pd.read_csv(path)
        targets = data.iloc[:, -1]
        inputs = data.iloc[:, :-1]
        # Para aproveitar a função de forward, salvamos os valores de entrada.
        old_inputs = self.inputs
        self.inputs = inputs
        res = self.__forward()
        # Restauramos os valores de entrada.
        self.inputs = old_inputs
        # Com o resultado salvo, retornamos uma matriz de confusão.
        # Antes, precisamos formatar o resultado para que ele seja compatível
        # com a coluna de classes do arquivo de teste.
        res = np.argmax(res, axis=1) + 1
        return confusion_matrix(targets, res)

    # Outras funções de set para o usuário.
    # Definir a função de ativação.
    def set_a_fun(self, a_fun):
        if a_fun == 'logistic':
            self.a_fun = logistic
            self.a_fun_d = logistic_derivative
        elif a_fun == 'tanh':
            self.a_fun = tanh
            self.a_fun_d = tanh_derivative

t = NeuralNetwork('logistic')
t.set_data('data/treinamento.csv')