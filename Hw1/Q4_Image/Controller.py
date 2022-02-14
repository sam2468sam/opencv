import sys
import cv2
import numpy

from HW1_4_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from scipy import signal

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
         super(QMainWindow, self).__init__(parent)
         self.setupUi(self)
         self.pushButton.clicked.connect(self.find)
         self.pushButton_1.clicked.connect(self.match)
         self.show()
         
     @pyqtSlot()
     def find(self):
          img1 = cv2.imread('Aerial1.jpg',0)
          img2 = cv2.imread('Aerial2.jpg',0)

          sift = cv2.xfeatures2d.SIFT_create()
          
          keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
          keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

          bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

          matches = bf.match(descriptors_1,descriptors_2)
          matches = sorted(matches, key = lambda x:x.distance)

          for i in range(6):
               img1 = cv2.drawKeypoints(img1,keypoints_1[matches[i].queryIdx:matches[i].queryIdx+1], img1,color = (0, 0, 255))
               img2 = cv2.drawKeypoints(img2,keypoints_2[matches[i].trainIdx:matches[i].trainIdx+1], img2,color = (0, 0, 255))
          #img_3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:6], None,matchColor = (0,0,255), flags = 2)

          cv2.imshow('my_image1', img1)
          cv2.imshow('my_image2', img2)
          cv2.imwrite('FeatureAerial1.jpg', img1)
          cv2.imwrite('FeatureAerial2.jpg', img2)
          #cv2.imshow('my_image3', img_3)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          
     def match(self):
          img1 = cv2.imread('Aerial1.jpg',0)
          img2 = cv2.imread('Aerial2.jpg',0)

          sift = cv2.xfeatures2d.SIFT_create()
          
          keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
          keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

          bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

          matches = bf.match(descriptors_1,descriptors_2)
          matches = sorted(matches, key = lambda x:x.distance)

          img_3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:6], None,matchColor = (0,0,255), flags = 2)

          cv2.imshow('my_image3', img_3)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
         
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())


