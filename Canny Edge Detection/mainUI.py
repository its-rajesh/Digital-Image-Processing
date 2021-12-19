import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import mycanny
import matplotlib.figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class userinterface(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.layout = QVBoxLayout(self)
        lbl1 = QLabel('Select an Image and switch the tabs for various results of Canny edge detection', self)
        lbl1.setAlignment(Qt.AlignCenter)

        btn1 = QPushButton('Select an Image', self)
        btn1.move(15, 40)

        self.layout.addWidget(lbl1)
        self.layout.addWidget(btn1)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Input Image")
        self.tabs.addTab(self.tab2, "Canny using OpenCV")
        self.tabs.addTab(self.tab3, "My Canny Algorithm")
        self.tabs.addTab(self.tab4, "Steps")

        # first tab
        self.tab1.layout = QVBoxLayout(self)
        self.tab1.setLayout(self.tab1.layout)

        # second tab
        self.tab2.layout = QVBoxLayout(self)
        self.tab2.setLayout(self.tab2.layout)

        # Third tab
        self.tab3.layout = QVBoxLayout(self)
        self.tab3.setLayout(self.tab3.layout)

        # fourth tab
        self.tab4.layout = QHBoxLayout(self)
        self.tab4.setLayout(self.tab4.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.setGeometry(600, 600, 1024, 720)
        self.setWindowTitle('Canny Edge Detection Algorithm')

        btn1.clicked.connect(self.open)
        self.show()

    def open(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open a file', '',
                                                'All Files (*.*)')
        if self.path != ('', ''):
            print("File path : " + self.path[0])

        try:
            self.compareAlgorithm(self.path[0])
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Error Loading an Image!!!")
            msg.setText("OOPS! \nAN ERROR OCCURED WHILE LOADING AN IMAGE\nPlease select an Image to proceed! (.jpg/.jpeg/.png)")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()

    def compareAlgorithm(self, path):
        # print(path)
        self.inputImage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        imagecheck = cv2.imread(path)
        imagecheck = cv2.cvtColor(imagecheck, cv2.COLOR_BGR2RGB)
        self.inputImage = np.array(self.inputImage, dtype=np.uint8)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        if imagecheck.shape[2] == 3:
        	sc.axes.imshow(imagecheck)
        else:
        	sc.axes.imshow(self.inputImage, cmap='gray')
        self.tab1.layout.addWidget(sc)

        img_canny = cv2.Canny(self.inputImage, 100, 200)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(img_canny, cmap='gray')
        self.tab2.layout.addWidget(sc)

        myresult = mycanny.main_function(self.inputImage)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(myresult, cmap='gray')
        self.tab3.layout.addWidget(sc)

        g, s, n, t, o = mycanny.return_all_images(self.inputImage)

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.tab4.layout.addWidget(self.canvas)
        self.figure.suptitle("Results of Each Step")

        ax1 = self.figure.add_subplot(231)
        ax1.imshow(self.inputImage, cmap='gray')
        ax1.set_title('Input')
        ax2 = self.figure.add_subplot(232)
        ax2.imshow(g, cmap='gray')
        ax2.set_title('Gaussian')
        ax3 = self.figure.add_subplot(233)
        ax3.imshow(s, cmap='gray')
        ax3.set_title('Sobel')
        ax4 = self.figure.add_subplot(234)
        ax4.imshow(n, cmap='gray')
        ax4.set_title('Non Max Suppression')
        ax5 = self.figure.add_subplot(235)
        ax5.imshow(t, cmap='gray')
        ax5.set_title('Thresholding')
        ax6 = self.figure.add_subplot(236)
        ax6.imshow(o, cmap='gray')
        ax6.set_title('Result')


def main():
    app = QApplication(sys.argv)
    ex = userinterface()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
