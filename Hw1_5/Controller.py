import sys
import cv2
import numpy
import random
import tensorflow
import os
import matplotlib.pyplot as plt

from HW1_5_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
         super(QMainWindow, self).__init__(parent)
         self.setupUi(self)
         self.pushButton.clicked.connect(self.show_image)
         self.pushButton_2.clicked.connect(self.show_hyperparameter)
         self.pushButton_3.clicked.connect(self.show_model)
         self.pushButton_4.clicked.connect(self.show_accuracy)
         self.pushButton_5.clicked.connect(self.test)
         self.show()
         
     @pyqtSlot()
     def show_image(self):
          label_dict={0:"plane",1:"car",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
          (trainX, trainY), (testX, testY) = cifar10.load_data()

          for i in range(10):
               index = random.randint(0,100)
               print(label_dict[trainY[index][0]])

               cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
               cv2.imshow('my_image', trainX[index])
               cv2.waitKey(0)
               cv2.destroyAllWindows()

     def show_hyperparameter(self):
          print('Hyperparameters: ')
          print('   Batch size: 128')
          print('   Learning rate: 0.01')
          print('   Optimizer: SGD')

     def show_model(self):

          model = tensorflow.keras.models.load_model("my_model.h5")
          model.summary()

     def show_accuracy(self):
          ACC = cv2.imread('Accuracy_rate.jpg')
          LOS = cv2.imread('loss.jpg')
          cv2.imshow('Accuracy_rate', ACC)
          cv2.imshow('loss', LOS)
          cv2.waitKey(0)
          cv2.destroyAllWindows()

     def test(self):
          index = self.lineEdit.text()
          
          label = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
          (trainX, trainY), (testX, testY) = cifar10.load_data()

          trainX_normalize=trainX.astype('float32')/255.0
          testX_normalize=testX.astype('float32')/255.0

          trainY_onehot=np_utils.to_categorical(trainY)
          testY_onehot=np_utils.to_categorical(testY)

          model = tensorflow.keras.models.load_model("my_model.h5")

          img = numpy.reshape(trainX_normalize[int(index)], (1, 32, 32, 3))
          ans = model.predict(img)
          # print(ans[0])

          fig = plt.figure()
          plt.xlabel("Label")
          plt.ylabel("Probability")
          plt.xticks(range(10))
          plt.bar(label,ans[0])
          plt.show()

          cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
          cv2.imshow('my_image', trainX[int(index)])
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          plt.close()
         
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())


