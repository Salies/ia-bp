from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableWidgetItem, QTableWidget, QMessageBox, QVBoxLayout, 
    QMainWindow, QPushButton, QGroupBox, QWidget, QHBoxLayout, QLabel, QSpinBox, QFileDialog, 
    QRadioButton, QDoubleSpinBox, QGridLayout
)
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import QUrl
from data_utils import import_data
from NeuralNetwork import NeuralNetwork

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projeto Rede Neural Backpropagation")
        self.centralWidget = QWidget(self)
        self.centralLayout = QVBoxLayout()
        self.createTrainingUpload()
        self.createTrainingTransFunction()
        self.createTrainingParameters()
        self.createTestArea()
        self.centralWidget.setLayout(self.centralLayout)
        self.setCentralWidget(self.centralWidget)

    def createTrainingUpload(self):
        # Criando os componentes base
        self.trainingUploadGroup = QGroupBox("Arquivo de treinamento")
        trainingLayout = QVBoxLayout(self.trainingUploadGroup)

        # Botão para abrir o arquivo de treinamento
        self.openTrainingFileButton = QPushButton("Abrir arquivo de treinamento")
        # Pass text "Abrir arquivo de treinamento" to openFile function
        self.openTrainingFileButton.clicked.connect(lambda: self.openFile("Abrir arquivo de treinamento"))

        # Label informando o arquivo selecionado
        self.trainingFileLabel = QLabel("Nenhum arquivo selecionado")
        self.trainingFileLabel.setWordWrap(True)

        # Label para indicar o número de neurônios na camada de entrada e saída
        self.neuronsLabel = QLabel()
        self.updateNeuronCount(0, 0)

        # Seleção do número de camadas ocultas
        hiddenLayersLayout = QHBoxLayout()
        hiddenLayersLabel = QLabel("Número de camadas ocultas:")
        self.hiddenLayersInput = QSpinBox()
        self.hiddenLayersInput.setMinimum(0)
        self.hiddenLayersInput.setMaximum(1000)
        self.hiddenLayersInput.setValue(0)
        hiddenLayersLayout.addWidget(hiddenLayersLabel)
        hiddenLayersLayout.addWidget(self.hiddenLayersInput)

        # Retornando
        trainingLayout.addWidget(self.trainingFileLabel)
        trainingLayout.addWidget(self.openTrainingFileButton)
        trainingLayout.addWidget(self.neuronsLabel)
        trainingLayout.addLayout(hiddenLayersLayout)
        self.centralLayout.addWidget(self.trainingUploadGroup)

    def createTrainingTransFunction(self):
        self.trainingTransFunctionGroup = QGroupBox("Função de transferência")
        trainingLayout = QVBoxLayout(self.trainingTransFunctionGroup)

        # Cria as opções
        logisticButton = QRadioButton("Logística")
        logisticButton.setChecked(True)
        tanhButton = QRadioButton("Tang. hiperbólica")

        # Adiciona as opções ao layout
        trainingLayout.addWidget(logisticButton)
        trainingLayout.addWidget(tanhButton)

        # Retornando
        self.centralLayout.addWidget(self.trainingTransFunctionGroup)

    def createTrainingParameters(self):
        # Criando os componentes base
        self.trainingParametersGroup = QGroupBox("Treinamento")
        trainingLayout = QGridLayout(self.trainingParametersGroup)

        errMaxRadio = QRadioButton("Erro máximo:")
        nItRadio = QRadioButton("Número de iterações:")
        errMaxRadio.setChecked(True)

        self.errMaxInput = QDoubleSpinBox()
        self.errMaxInput.setMinimum(0.0)
        self.errMaxInput.setMaximum(100.0)
        self.errMaxInput.setValue(0.01)

        self.nItInput = QSpinBox()
        self.nItInput.setMinimum(0)
        self.nItInput.setMaximum(100000)
        self.nItInput.setValue(1000)

        trainingLayout.addWidget(errMaxRadio, 0, 0)
        trainingLayout.addWidget(self.errMaxInput, 0, 1)
        trainingLayout.addWidget(nItRadio, 1, 0)
        trainingLayout.addWidget(self.nItInput, 1, 1)

        # Botão para iniciar o treinamento
        self.startTrainingButton = QPushButton("Treinar")
        self.startTrainingButton.clicked.connect(self.startTraining)
        trainingLayout.addWidget(self.startTrainingButton, 2, 0, 1, 2)

        # Retornando
        self.centralLayout.addWidget(self.trainingParametersGroup)

    def updateNeuronCount(self, inputNeurons, outputNeurons):
        inputNeuronsTxt = "A camada de <u>entrada</u> possui <b>{}</b> neurônios.".format(inputNeurons)
        outputNeuronsTxt = "A camada de <u>saída</u> possui <b>{}</b> neurônios.".format(outputNeurons)
        self.neuronsLabel.setText("{}<br/>{}".format(inputNeuronsTxt, outputNeuronsTxt))

    def openFile(self, title):
        # Abrindo o arquivo de treinamento
        file = QFileDialog.getOpenFileName(self, title, "", "Arquivos (*.csv)")

        # Verificando se há um arquivo selecionado
        if file[0] == "":
            return

        # Pegando o nome do arquivo
        fileName = QUrl(file[0]).fileName()

        if title == "Abrir arquivo de treinamento":
            self.trainingData = import_data(file[0])
            self.trainingFileLabel.setText(fileName)
            self.updateNeuronCount(self.trainingData[2], self.trainingData[3])
            self.hiddenLayersInput.setValue(self.trainingData[4])
            return

        self.testFilePath = file[0]
        self.testFileLabel.setText(fileName)

    def startTraining(self):
        if(not hasattr(self, 'trainingData')):
            QMessageBox.critical(self, "Erro!", "Nenhum arquivo de treinamento selecionado.")
            return

        transFunctions = ['logistic', 'tanh']
        stopCriterias = ['error', 'epochs']

        # Pega a função e o critério de parada selecionados
        checkedTransFunction = [button.isChecked() for button in self.trainingTransFunctionGroup.findChildren(QRadioButton)].index(True)
        checkedStopCriteria = [button.isChecked() for button in self.trainingParametersGroup.findChildren(QRadioButton)].index(True)

        # Dados para a rede neural
        inputs, targets, input_size, output_size, n_hidden = self.trainingData

        # Se o usuário escolheu um novo número de camadas ocultas, atualiza o valor
        if n_hidden != self.hiddenLayersInput.value():
            n_hidden = self.hiddenLayersInput.value()

        stopCriteria = stopCriterias[checkedStopCriteria]
        transFunction = transFunctions[checkedTransFunction]

        # Cria a rede neural
        self.nn = NeuralNetwork(inputs, targets, input_size, output_size, n_hidden, stopCriteria, transFunction)

        # Pega o valor de treinamento de acordo com o critério escolhido pelo usuário
        stopValue = (self.errMaxInput.value() if stopCriteria == 'error' else self.nItInput.value())

        # Treina a rede neural
        self.nn.train(stopValue)

        # Avisa quando o treinamento acabar
        QMessageBox.information(self, "Atenção!", "Treinamento concluído.")

    # Criação da área de teste
    def createTestArea(self):
        group = QGroupBox("Teste")
        layout = QVBoxLayout(group)

        # Label pro arquivo de teste
        self.testFileLabel = QLabel("Nenhum arquivo selecionado")
        self.testFileLabel.setWordWrap(True)

        # Botão para abrir o arquivo de teste
        self.openTestFileButton = QPushButton("Abrir arquivo de teste")
        self.openTestFileButton.clicked.connect(lambda: self.openFile("Abrir arquivo de teste"))

        # Criando a tabela para a matriz de confusão
        self.confusionMatrixTable = QTableWidget()
        # Tamanho automático
        self.confusionMatrixTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.confusionMatrixTable.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Desabilitando edição
        self.confusionMatrixTable.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Criando o botão para iniciar o teste
        self.startTestButton = QPushButton("Testar")
        self.startTestButton.clicked.connect(self.test)

        # Adicionando os componentes ao layout
        layout.addWidget(self.testFileLabel)
        layout.addWidget(self.openTestFileButton)
        layout.addWidget(self.confusionMatrixTable)
        layout.addWidget(self.startTestButton)

        # Retornando
        self.centralLayout.addWidget(group)

    def test(self):
        if(not hasattr(self, 'nn')):
            QMessageBox.critical(self, "Erro!", "A rede não está treinada.")
            return

        if(not hasattr(self, 'testFilePath')):
            QMessageBox.critical(self, "Erro!", "Nenhum arquivo de teste selecionado.")
            return

        cm = self.nn.test(self.testFilePath)

        # Atualizando a tabela
        self.confusionMatrixTable.setRowCount(cm.shape[0])
        self.confusionMatrixTable.setColumnCount(cm.shape[1])
        self.confusionMatrixTable.setHorizontalHeaderLabels([str(i + 1) for i in range(cm.shape[1])])
        self.confusionMatrixTable.setVerticalHeaderLabels([str(i + 1) for i in range(cm.shape[0])])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.confusionMatrixTable.setItem(i, j, QTableWidgetItem(str(cm[i, j])))