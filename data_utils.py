# Funções auxiliares para manipulação de dados.
# Abrem o conjunto de treinamento, vê a quantidade de entradas e saídas,
# estima o número de neurônios na camada oculta e embaralham os dados.

import pandas as pd
import numpy as np

SEED = 666

def import_data(data_path):
    data = pd.read_csv(data_path)
    # Embaralha os dados para que o treinamento seja mais eficiente.
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # A última coluna é o target
    targets = data.iloc[:, -1].values.astype(int)
    output_size = len(np.unique(targets))
    inputs = data.iloc[:, :-1].values
    input_size = inputs.shape[1]

    return inputs, targets, input_size, output_size