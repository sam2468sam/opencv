import sys
import cv2
import numpy
import glob

from HW1_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from scipy import signal

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
         super(QMainWindow, self).__init__(parent)
         self.setupUi(self)
         self.pushButton.clicked.connect(self.find_corners)
         self.pushButton_2.clicked.connect(self.find_intrinsic)
         self.pushButton_3.clicked.connect(self.find_extrinsic)
         self.pushButton_4.clicked.connect(self.find_distortion)
         self.show()
         self.ret=0
         self.mtx=0
         self.dist=0
         self.rvecs=0
         self.tvecs=0
         
     @pyqtSlot()
     def find_corners(self):
         # termination criteria
         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

         # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
         objp = numpy.zeros((11 * 8, 3), numpy.float32)
         objp[:, : 2] = numpy.mgrid[0: 11, 0: 8].T.reshape(-1, 2)

         # Arrays to store object points and image points from all the images.
         objpoints = [] # 3d point in real world space
         imgpoints = [] # 2d points in image plane
         
         images = glob.glob('*.bmp')

         for fname in images:
              img = cv2.imread(fname)
              gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

              # Find the chess board corners
              ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

              # If found, add object points, image points (after refining them)
              if ret == True:
                   objpoints.append(objp)

                   corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                   imgpoints.append(corners2)
                   cv2.drawChessboardCorners(img, (11, 8), corners2, ret)

                   cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
                   cv2.imshow('my_image', img)
                   cv2.waitKey(0)

         cv2.destroyAllWindows()

         self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
         
     def find_intrinsic(self):
         print(self.mtx)
     def find_extrinsic(self):
         # termination criteria
         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

         # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
         objp = numpy.zeros((11 * 8, 3), numpy.float32)
         objp[:, : 2] = numpy.mgrid[0: 11, 0: 8].T.reshape(-1, 2)

         # Arrays to store object points and image points from all the images.
         objpoints = [] # 3d point in real world space
         imgpoints = [] # 2d points in image plane
         
         text = self.comboBox.currentText()
         img = cv2.imread(text+'.bmp')
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

         ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
         objpoints.append(objp)
         imgpoints.append(corners)

         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
         
         rotation = rvecs.pop()
         transition = tvecs.pop()
         m_rotation , jacobin = cv2.Rodrigues(rotation)
         extrinsic = numpy.hstack((m_rotation,transition))
         print(extrinsic)
         
     def find_distortion(self):
         print(self.dist)
         
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())


