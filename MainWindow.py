# PySide6 main window example

import sys
from PySide6.QtWidgets import QVBoxLayout, QMainWindow, QPushButton, QGroupBox, QWidget, QHBoxLayout, QLabel, QSpinBox, QFileDialog, QRadioButton
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import QUrl

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projeto Rede Neural Backpropagation")
        self.centralWidget = QWidget(self)
        self.centralLayout = QVBoxLayout()
        self.createTrainingUpload()
        self.createTrainingTransFunction()
        self.centralWidget.setLayout(self.centralLayout)
        self.setCentralWidget(self.centralWidget)

    def createTrainingUpload(self):
        # Criando os componentes base
        trainingGroup = QGroupBox("Arquivo de treinamento")
        trainingLayout = QVBoxLayout(trainingGroup)

        # Botão para abrir o arquivo de treinamento
        self.openTrainingFileButton = QPushButton("Abrir arquivo de treinamento")
        # Pass text "Abrir arquivo de treinamento" to openFile function
        self.openTrainingFileButton.clicked.connect(lambda: self.openFile("Abrir arquivo de treinamento"))

        # Label informando o arquivo selecionado
        self.trainingFileLabel = QLabel("Nenhum arquivo selecionado")

        # Label para indicar o número de neurônios na camada de entrada e saída
        inputNeuronsTxt = "A camada de <u>entrada</u> possui <b>0</b> neurônios."
        outputNeuronsTxt = "A camada de <u>saída</u> possui <b>0</b> neurônios."
        self.neuronsLabel = QLabel("%s<br/>%s " % (inputNeuronsTxt, outputNeuronsTxt))

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
        self.centralLayout.addWidget(trainingGroup)

    def createTrainingTransFunction(self):
        # Criando os componentes base
        trainingGroup = QGroupBox("Função de transferência")
        trainingLayout = QVBoxLayout(trainingGroup)

        # Cria as opções
        logisticButton = QRadioButton("Logística")
        logisticButton.setChecked(True)
        tanhButton = QRadioButton("Tang. hiperbólica")

        # Adiciona as opções ao layout
        trainingLayout.addWidget(logisticButton)
        trainingLayout.addWidget(tanhButton)

        # Retornando
        self.centralLayout.addWidget(trainingGroup)

    def openFile(self, title):
        # Abrindo o arquivo de treinamento
        file = QFileDialog.getOpenFileName(self, title, "", "Arquivos (*.csv)")

        # Verificando se há um arquivo selecionado
        if file[0] == "":
            return

        # Pegando o nome do arquivo
        fileName = QUrl(file[0]).fileName()

        if title == "Abrir arquivo de treinamento":
            self.trainingFile = file
            self.trainingFileLabel.setText(fileName)
            return