# Arquivo para definir as funções de ativação, erro, e demias funções utilizadas pela rede neural.

import numpy as np

# Tangente hiperbólica
def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

# Função logística
# Aqui utilizamos uma versão normalizada da função logística, que retorna valores entre -1 e 1.
def logistic(x):
  return np.exp(-np.logaddexp(-x, 0)) * 2 - 1

def logistic_derivative(x):
  return 0.5 * (1 + logistic(x)) * (1 - logistic(x))

# Função de erro proposta pelo trabalho
def error_function(targets, outputs):
    return np.sum((targets - outputs)**2) / 2

# Prepara os targets para uso na rede
# A lógica é: se o target é 1, então a saída da rede deve ser [1, -1, -1, -1, -1]
# Se o target é 2, então a saída da rede deve ser [-1, 1, -1, -1, -1]
# E assim por diante.
def prepare_targets(targets, n_outputs):
    n = targets.shape[0]
    # Aqui fazemos isso de um jeito "esperto"
    # Primeiro, criamos uma matriz com o mesmo número de linhas que os targets, mas com n_outputs colunas
    # (n_outputs é o número de classes).
    # Em seguida, preenchemos a matriz com -1.
    # Depois, para cada linha, colocamos 1 na coluna que corresponde ao target.
    targets_matrix = np.full((n, n_outputs), fill_value=-1)
    # Os targets vão de 1 a n_outputs, mas os índices vão de 0 a n_outputs - 1, então subtraímos 1.
    targets_matrix[np.arange(n), targets - 1] = 1
    return targets_matrix

def confusion_matrix(y_true, y_pred):
    # Infere o número de classes
    n_classes = len(np.unique(y_true))
    # Cria a matriz de confusão
    cm = np.zeros((n_classes, n_classes), dtype=int)
    # Para cada classe verdadeira
    for i in range(n_classes):
        # Para cada classe predita
        for j in range(n_classes):
            # Conta quantas vezes a classe verdadeira i foi predita como j
            cm[i, j] = np.sum(np.logical_and(y_true == i, y_pred == j))
    return cm