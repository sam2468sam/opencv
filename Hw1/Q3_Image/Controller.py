import sys
import cv2
import numpy

from HW1_3_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from scipy import signal

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
          super(QMainWindow, self).__init__(parent)
          self.setupUi(self)
          self.pushButton.clicked.connect(self.disparity)
          self.show()
         
     @pyqtSlot()
     def disparity(self):
          imgL = cv2.imread('imL.png',0)
          imgR = cv2.imread('imR.png',0)

          stereo = cv2.StereoBM_create(16, 15)
          disparity = stereo.compute(imgL,imgR)
          disparity = cv2.convertScaleAbs(disparity)

          cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
          cv2.imshow('my_image', disparity)
          cv2.waitKey(0)

          cv2.destroyAllWindows()
          
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())


