# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menumatbert_model = QtWidgets.QMenu(self.menubar)
        self.menumatbert_model.setObjectName("menumatbert_model")
        self.menuSave_path = QtWidgets.QMenu(self.menubar)
        self.menuSave_path.setObjectName("menuSave_path")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionmodel_umap = QtWidgets.QAction(MainWindow)
        self.actionmodel_umap.setObjectName("actionmodel_umap")
        self.actionmodel_plot = QtWidgets.QAction(MainWindow)
        self.actionmodel_plot.setObjectName("actionmodel_plot")
        self.actioncosine_similarity = QtWidgets.QAction(MainWindow)
        self.actioncosine_similarity.setObjectName("actioncosine_similarity")
        self.actionSave_path = QtWidgets.QAction(MainWindow)
        self.actionSave_path.setObjectName("actionSave_path")
        self.actionmodel_path = QtWidgets.QAction(MainWindow)
        self.actionmodel_path.setObjectName("actionmodel_path")
        self.menumatbert_model.addAction(self.actionmodel_path)
        self.menumatbert_model.addAction(self.actionmodel_umap)
        self.menumatbert_model.addAction(self.actionmodel_plot)
        self.menumatbert_model.addAction(self.actioncosine_similarity)
        self.menuSave_path.addAction(self.actionSave_path)
        self.menubar.addAction(self.menuSave_path.menuAction())
        self.menubar.addAction(self.menumatbert_model.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menumatbert_model.setTitle(_translate("MainWindow", "matbert-model "))
        self.menuSave_path.setTitle(_translate("MainWindow", "Save path"))
        self.actionmodel_umap.setText(_translate("MainWindow", "umap"))
        self.actionmodel_plot.setText(_translate("MainWindow", "plot"))
        self.actioncosine_similarity.setText(_translate("MainWindow", "cosine similarity ranking"))
        self.actionSave_path.setText(_translate("MainWindow", "Save path"))
        self.actionmodel_path.setText(_translate("MainWindow", "model path"))
