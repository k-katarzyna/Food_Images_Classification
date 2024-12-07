from sys import exit
from os import path
from PyQt5 import QtCore, QtGui, QtWidgets

class customGUI(object):
    photoPath = ""

    def setupUi(self, Window):
        self.Window = Window
        self.windowGrid = QtWidgets.QGridLayout(Window)
        self.windowGrid.setObjectName("windowGrid")
        
        self.imageLabel = QtWidgets.QLabel(Window)
        self.imageLabel.setObjectName("imageLabel")
        self.windowGrid.addWidget(self.imageLabel, 0, 0, 1, 1)

        self.photoPath = path.dirname(__file__) + "/../noPhoto.png"
        self.updateImage()
        
        self.analyzeButton = QtWidgets.QPushButton(Window)
        self.analyzeButton.setObjectName("analyzeButton")
        self.analyzeButton.setText("Analyze Photo")
        self.windowGrid.addWidget(self.analyzeButton, 2, 0, 1, 1)
        
        self.importButton = QtWidgets.QPushButton(Window)
        self.importButton.setObjectName("importButton")
        self.importButton.setText("Import Photo")
        self.windowGrid.addWidget(self.importButton, 1, 0, 1, 1)

        QtCore.QMetaObject.connectSlotsByName(Window)

    def updateImage(self):
        imagePixmap = QtGui.QPixmap(self.photoPath)
        self.imageLabel.setPixmap(imagePixmap)
        self.Window.resize(imagePixmap.width(), imagePixmap.height() + 100)


    def analyzeButtonFunction(self):
        print("analyze")


    def importButtonFunction(self):
        result = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            'Open file', 
            'c:\\',
            "Image files (*.jpg *.png)"
        )
        self.photoPath = result[0]
        self.updateImage()

    def connectSignals(self):
        self.analyzeButton.clicked.connect(self.analyzeButtonFunction)
        self.importButton.clicked.connect(self.importButtonFunction)




class CustomWindow(QtWidgets.QWidget, customGUI):
    def __init__(self, parent = None,):
        super().__init__(parent)
        # self.setObjectName("rootWindow")
        self.setWindowTitle("Food Detector")
        # self.resize(300, 400)
        self.setupUi(self)
        self.connectSignals()


def run_app():
  app = QtWidgets.QApplication([])
  win = CustomWindow()
  win.show()
  exit(app.exec_()) # close python script on app exit


if __name__ == '__main__':
    run_app()
