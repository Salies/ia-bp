'''
    Trabalho de Inteligência Artificial
    Backpropagation
    Alunos:
        - Daniel Henrique Serezane Pereira
        - Guilherme Cesar Tomiasi
'''

from MainWindow import MainWindow
from PySide6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())