from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QMainWindow, QPushButton, QGroupBox, QWidget, QHBoxLayout, QLabel, QSpinBox, QFileDialog, QRadioButton, QDoubleSpinBox, QGridLayout
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import QUrl
from data_utils import import_data

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projeto Rede Neural Backpropagation")
        self.centralWidget = QWidget(self)
        self.centralLayout = QVBoxLayout()
        self.createTrainingUpload()
        self.createTrainingTransFunction()
        self.createTrainingParameters()
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

    def startTraining(self):
        if(not hasattr(self, 'trainingData')):
            QMessageBox.critical(self, "Erro!", "Nenhum arquivo de treinamento selecionado.")
            return

        transFunction = ['logistic', 'tanh']
        stopCriteria = ['error', 'epochs']

        # index of checked button
        checkedTransFunction = [button.isChecked() for button in self.trainingTransFunctionGroup.findChildren(QRadioButton)].index(True)
        checkedStopCriteria = [button.isChecked() for button in self.trainingParametersGroup.findChildren(QRadioButton)].index(True)

        print()
        print(transFunction[checkedTransFunction])
        print(stopCriteria[checkedStopCriteria])
        print()

        inputs, targets, input_size, output_size, n_hidden = self.trainingData