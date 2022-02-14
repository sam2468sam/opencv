import sys
import cv2
import numpy
import glob
import time

from HW1_2_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from scipy import signal

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
         super(QMainWindow, self).__init__(parent)
         self.setupUi(self)
         self.pushButton.clicked.connect(self.tetrahedron)
         self.show()
         
     @pyqtSlot()
     def tetrahedron(self):
         # termination criteria
         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

         # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
         objp = numpy.zeros((11 * 8, 3), numpy.float32)
         objp[:, : 2] = numpy.mgrid[0: 11, 0: 8].T.reshape(-1, 2)

         # Arrays to store object points and image points from all the images.
         objpoints = [] # 3d point in real world space
         imgpoints = [] # 2d points in image plane
         axis = numpy.float32([[3,3,-3],[1,1,0],[3,5,0],[5,1,0]])
         
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

                   ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

                   objpoints.pop()
                   imgpoints.pop()
                   rotation = rvecs.pop()
                   transition = tvecs.pop()
                   
                   # project 3D points to image plane
                   imgpts, jac = cv2.projectPoints(axis, rotation, transition, mtx, dist)

                   img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 4)
                   img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 4)
                   img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 4)
                   img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 4)
                   img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 4)
                   img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 4)

                   cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
                   cv2.imshow('my_image', img)
                   cv2.waitKey(500)

         cv2.destroyAllWindows()
         
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())


