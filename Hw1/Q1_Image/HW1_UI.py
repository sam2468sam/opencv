# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW1_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(349, 214)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 310, 180))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(20, 130, 100, 30))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 80, 100, 30))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(20, 30, 100, 30))
        self.pushButton.setObjectName("pushButton")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(150, 30, 151, 130))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(30, 20, 111, 20))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(10, 10))
        self.label_3.setSizeIncrement(QtCore.QSize(10, 10))
        self.label_3.setBaseSize(QtCore.QSize(10, 10))
        self.label_3.setObjectName("label_3")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setGeometry(QtCore.QRect(30, 50, 60, 20))
        self.comboBox.setEditable(True)
        self.comboBox.setMaxVisibleItems(15)
        self.comboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 90, 100, 30))
        self.pushButton_3.setObjectName("pushButton_3")

        self.retranslateUi(Form)
        self.comboBox.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "1. Calibration"))
        self.pushButton_4.setText(_translate("Form", "1.4 Find Distortion"))
        self.pushButton_2.setText(_translate("Form", "1.2 Find Intrinsic"))
        self.pushButton.setText(_translate("Form", "1.1 Find Corners"))
        self.groupBox_2.setTitle(_translate("Form", "1.3 Find Extrinsic"))
        self.label_3.setText(_translate("Form", "<html><head/><body><p>Select image</p></body></html>"))
        self.comboBox.setItemText(0, _translate("Form", "1"))
        self.comboBox.setItemText(1, _translate("Form", "2"))
        self.comboBox.setItemText(2, _translate("Form", "3"))
        self.comboBox.setItemText(3, _translate("Form", "4"))
        self.comboBox.setItemText(4, _translate("Form", "5"))
        self.comboBox.setItemText(5, _translate("Form", "6"))
        self.comboBox.setItemText(6, _translate("Form", "7"))
        self.comboBox.setItemText(7, _translate("Form", "8"))
        self.comboBox.setItemText(8, _translate("Form", "9"))
        self.comboBox.setItemText(9, _translate("Form", "10"))
        self.comboBox.setItemText(10, _translate("Form", "11"))
        self.comboBox.setItemText(11, _translate("Form", "12"))
        self.comboBox.setItemText(12, _translate("Form", "13"))
        self.comboBox.setItemText(13, _translate("Form", "14"))
        self.comboBox.setItemText(14, _translate("Form", "15"))
        self.pushButton_3.setText(_translate("Form", "1.3 Find Extrinsic"))
