import win32gui
import win32con
import sys
import os
import subprocess
import time
import win32gui
import PySide2
import PyQt5
import open3d as o3
import numpy as np
import time
import importlib
import torch
import threading
import _thread
import random
import math
import pandas as pd
from pathlib import Path
from multiprocessing import Process
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2 import QtWidgets,QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QStyleFactory

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # self.thread = None
        self.timer1 = QTimer()
        self.timer2 = QTimer()
        self.timer3 = QTimer()
        self.timer4 = QTimer()
        ui.timer1.timeout.connect(checkload)
        ui.timer2.timeout.connect(checkpre)
        ui.timer3.timeout.connect(checkheat)
        ui.timer4.timeout.connect(checkfeat)

        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setGeometry(QRect(100, 100, 1400, 850))
        MainWindow.setMaximumSize(1400, 850)
        MainWindow.setMinimumSize(1400, 850)

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(910, 10, 450, 180))
        self.textEdit.setReadOnly(True)
        self.textEdit.setStyleSheet('''
                                    QTextEdit {
                                        border: 2px solid gray;
                                        border-radius: 10px;
                                        padding: 0 8px;
                                        background: white;
                                        selection-background-color: darkgray;
                                    }
                                    ''')

        self.toolButton = QToolButton(self.centralwidget)
        self.toolButton.setObjectName(u"toolButton")
        self.toolButton.setGeometry(QRect(810, 50, 70, 30))
        self.toolButton.setStyleSheet('''
                                        QToolButton {
                                            border: 2px solid #8f8f91;
                                            border-radius: 6px;
                                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #f6f7fa, stop: 1 #dadbde);
                                        }
                                      ''')
        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(200, 50, 600, 30))
        self.lineEdit.setInputMethodHints(Qt.ImhMultiLine)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(80, 50, 120, 30))
        self.label.setFont(QFont("YouYuan",12))
        self.lineEdit.setStyleSheet('''
                                    QLineEdit {
                                        border: 2px solid gray;
                                        border-radius: 10px;
                                        padding: 0 8px;
                                        background: white;
                                        selection-background-color: darkgray;
                                    }
                                    ''')

        self.toolButton_1 = QToolButton(self.centralwidget)
        self.toolButton_1.setObjectName(u"toolButton_1")
        self.toolButton_1.setGeometry(QRect(810, 100, 70, 30))
        self.toolButton_1.setStyleSheet('''
                                        QToolButton {
                                            border: 2px solid #8f8f91;
                                            border-radius: 6px;
                                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #f6f7fa, stop: 1 #dadbde);
                                        }
                                      ''')
        self.lineEdit_1 = QLineEdit(self.centralwidget)
        self.lineEdit_1.setObjectName(u"lineEdit_1")
        self.lineEdit_1.setGeometry(QRect(200, 100, 600, 30))
        self.lineEdit_1.setInputMethodHints(Qt.ImhMultiLine)
        self.label_1 = QLabel(self.centralwidget)
        self.label_1.setObjectName(u"label")
        self.label_1.setGeometry(QRect(80, 100, 120, 30))
        self.label_1.setFont(QFont("YouYuan",12))

        self.lineEdit_1.setStyleSheet('''
                                    QLineEdit {
                                        border: 2px solid gray;
                                        border-radius: 10px;
                                        padding: 0 8px;
                                        background: white;
                                        selection-background-color: darkgray;
                                    }
                                    ''')

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(200, 140, 600, 80))
        self.groupBox.setStyleSheet("border:none")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")

        self.radioButton = QRadioButton(self.groupBox)
        self.radioButton.setObjectName(u"radioButton")
        self.gridLayout.addWidget(self.radioButton, 0, 0, 1, 1)
        self.radioButton.setFont(QFont("YouYuan",12))
        self.radioButton.setChecked(True)

        self.radioButton_2 = QRadioButton(self.groupBox)
        self.radioButton_2.setObjectName(u"radioButton_2")
        self.gridLayout.addWidget(self.radioButton_2, 0, 1, 1, 1)
        self.radioButton_2.setFont(QFont("YouYuan",12))

        self.radioButton_3 = QRadioButton(self.groupBox)
        self.radioButton_3.setObjectName(u"radioButton_3")
        self.gridLayout.addWidget(self.radioButton_3, 0, 2, 1, 1)
        self.radioButton_3.setFont(QFont("YouYuan",12))

        self.radioButton_4 = QRadioButton(self.groupBox)
        self.radioButton_4.setObjectName(u"radioButton_4")
        self.gridLayout.addWidget(self.radioButton_4, 0, 3, 1, 1)
        self.radioButton_4.setFont(QFont("YouYuan",12))

        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(80, 230, 400, 200))
        self.formLayout = QFormLayout(self.widget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, 0, 0)

        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")
        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.pushButton)
        self.pushButton.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : white;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')

        self.pushButton_2 = QPushButton(self.widget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.pushButton_2)
        self.pushButton_2.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : white;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')
    

        self.pushButton_3 = QPushButton(self.widget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.pushButton_3)
        self.pushButton_3.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : white;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')

        self.pushButton_4 = QPushButton(self.widget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.pushButton_4)
        self.pushButton_4.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : white;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')

        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(200, 250, 1100, 550))

        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(200, 230, 1100, 70))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)

        self.label_p = QLabel(self.centralwidget)
        self.label_p.setObjectName(u"label_p")
        self.label_p.setGeometry(QRect(220, 245, 120, 40))
        self.label_p.setFont(QFont("YouYuan",12))

        self.progressBar = QProgressBar(self.centralwidget)
        self.progressBar.setStyleSheet('''
                                        #GreenProgressBar {
                                            min-height: 12px;
                                            max-height: 12px;
                                            order-radius: 6px;
                                        }
                                        ''')
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(350, 245, 850, 40))
        self.progressBar.setValue(0)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

        

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"牙冠特征提取可视化", None))
        self.label_p.setText(QCoreApplication.translate("MainWindow", u"Progress", None))
        self.toolButton.setText(QCoreApplication.translate("MainWindow", u"选择文件", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"牙 冠 文 件", None))
        self.toolButton_1.setText(QCoreApplication.translate("MainWindow", u"选择文件", None))
        self.label_1.setText(QCoreApplication.translate("MainWindow", u"特 征 文 件", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u" 原始点云 ", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u" 热图点云 ", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"预测特征点", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"实际特征点", None))
        self.radioButton.setText(QCoreApplication.translate("MainWindow", u"牙切点CO", None))
        self.radioButton_2.setText(QCoreApplication.translate("MainWindow", u"牙尖点CU", None))
        self.radioButton_3.setText(QCoreApplication.translate("MainWindow", u"面轴点FA", None))
        self.radioButton_4.setText(QCoreApplication.translate("MainWindow", u"咬合点OC", None))

    # retranslateUi

TRAINED = False
vis = None
def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd]
    vis.create_window('Point Cloud Visualization', 1100, 550, 200, 250, False)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def open_file_xyz(filepath):
    if filepath == '':
        QMessageBox.information(None, '提示','未选择点云文件',QMessageBox.Yes)
        return
    if not filepath.endswith('.txt'):
        QMessageBox.information(None, '提示','点云文件非法',QMessageBox.Yes)
        return
    hwnd = win32gui.FindWindow(None , 'Point Cloud Visualization')
    if hwnd != 0: 
        global vis
        vis.destroy_window()
    vis = o3.visualization.Visualizer()
    source = o3.io.read_point_cloud(filename = str(filepath),format = "xyz")
    source.paint_uniform_color([0.8,0.8,0.8])
    custom_draw_geometry(source)

def open_file_heat(filepath):
    featurename = ['-heatCO' , '-heatCU' , '-heatFA' , '-heatOC']

    global TRAINED
    if TRAINED == False:

        ui.textEdit.append("<font color=\"#FF0000\">ERROR : 点云数据没有在模型中进行预测</font>")

        return

    if ui.radioButton.isChecked():
        feature = 0
    elif ui.radioButton_2.isChecked():
        feature = 1
    elif ui.radioButton_3.isChecked():
        feature = 2
    elif ui.radioButton_4.isChecked():
        feature = 3
    else:
        feature = -1
    
    if feature == -1:
        return

    hwnd = win32gui.FindWindow(None , 'Point Cloud Visualization')
    if hwnd != 0: 
        global vis
        vis.destroy_window()
    vis = o3.visualization.Visualizer()

    filename = os.path.basename(filepath)
    filename = filename.split(".")[0]  + featurename[feature] + '.txt'
    filepath = './Visualization/pred_data/' + filename
    
    source = o3.io.read_point_cloud(filename = str(filepath),format = "xyzrgb")
    custom_draw_geometry(source)

def draw_pre_feature(filepath):
    '''
    显示出预测的特征点
    '''
    featurename = ['-PredCO' , '-PredCU' , '-PredFA' , '-PredOC']

    global TRAINED
    if TRAINED == False:

        ui.textEdit.append("<font color=\"#FF0000\">ERROR : 点云数据没有在模型中进行预测</font>")

        return

    if ui.radioButton.isChecked():
        feature = 0
    elif ui.radioButton_2.isChecked():
        feature = 1
    elif ui.radioButton_3.isChecked():
        feature = 2
    elif ui.radioButton_4.isChecked():
        feature = 3
    else:
        feature = -1
    
    if feature == -1:
        return

    hwnd = win32gui.FindWindow(None , 'Point Cloud Visualization')
    if hwnd != 0: 
        global vis
        vis.destroy_window()
    vis = o3.visualization.Visualizer()

    filename = os.path.basename(filepath)
    filename = filename.split(".")[0]  + featurename[feature] + '.txt'
    filepath = './Visualization/pred_data/' + filename
    
    source = o3.io.read_point_cloud(filename = str(filepath),format = "xyzrgb")
    custom_draw_geometry(source)

def duplif(ncent , cent_points):
    cent_eh = np.zeros(shape = (ncent * 24 , 6))
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + i*ncent][0] = cent_points[j][0]
            cent_eh[j + i*ncent][1] = cent_points[j][1]
            cent_eh[j + i*ncent][2] = cent_points[j][2]
            cent_eh[j + i*ncent][i] = cent_points[j][i] + 0.1
            cent_eh[j + i*ncent][3] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (3 + i)*ncent][0] = cent_points[j][0] 
            cent_eh[j + (3 + i)*ncent][1] = cent_points[j][1] 
            cent_eh[j + (3 + i)*ncent][2] = cent_points[j][2] 
            cent_eh[j + (3 + i)*ncent][i] = cent_points[j][i] - 0.1
            cent_eh[j + (3 + i)*ncent][3] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (6 + i)*ncent][0] = cent_points[j][0] + 0.1
            cent_eh[j + (6 + i)*ncent][1] = cent_points[j][1] + 0.1
            cent_eh[j + (6 + i)*ncent][2] = cent_points[j][2] + 0.1
            cent_eh[j + (6 + i)*ncent][i] = cent_points[j][i] - 0.1
            cent_eh[j + (6 + i)*ncent][3] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (9 + i)*ncent][0] = cent_points[j][0] - 0.1
            cent_eh[j + (9 + i)*ncent][1] = cent_points[j][1] - 0.1
            cent_eh[j + (9 + i)*ncent][2] = cent_points[j][2] - 0.1
            cent_eh[j + (9 + i)*ncent][i] = cent_points[j][i] + 0.1
            cent_eh[j + (9 + i)*ncent][3] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (12 + i)*ncent][0] = cent_points[j][0]
            cent_eh[j + (12 + i)*ncent][1] = cent_points[j][1]
            cent_eh[j + (12 + i)*ncent][2] = cent_points[j][2]
            cent_eh[j + (12 + i)*ncent][i] = cent_points[j][i] + 0.05
            cent_eh[j + (12 + i)*ncent][3] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (15 + i)*ncent][0] = cent_points[j][0] 
            cent_eh[j + (15 + i)*ncent][1] = cent_points[j][1] 
            cent_eh[j + (15 + i)*ncent][2] = cent_points[j][2] 
            cent_eh[j + (15 + i)*ncent][i] = cent_points[j][i] - 0.05
            cent_eh[j + (15 + i)*ncent][3] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (18 + i)*ncent][0] = cent_points[j][0] + 0.05
            cent_eh[j + (18 + i)*ncent][1] = cent_points[j][1] + 0.05
            cent_eh[j + (18 + i)*ncent][2] = cent_points[j][2] + 0.05
            cent_eh[j + (18 + i)*ncent][i] = cent_points[j][i] - 0.05
            cent_eh[j + (18 + i)*ncent][3] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (21 + i)*ncent][0] = cent_points[j][0] - 0.05
            cent_eh[j + (21 + i)*ncent][1] = cent_points[j][1] - 0.05
            cent_eh[j + (21 + i)*ncent][2] = cent_points[j][2] - 0.05
            cent_eh[j + (21 + i)*ncent][i] = cent_points[j][i] + 0.05
            cent_eh[j + (21 + i)*ncent][3] = 1
    return cent_eh

def draw_feature(filepath , featfilepath):
    '''
    显示出原有的特征点和预测的特征点
    '''
    feat = ['CO.txt' , 'CU.txt' , 'FA.txt' , 'OC.txt']
    featurename = ['-PredCO' , '-PredCU' , '-PredFA' , '-PredOC']
    featurenameW = ['-PredCOW' , '-PredCUW' , '-PredFAW' , '-PredOCW']

    global TRAINED
    if TRAINED == False:

        ui.textEdit.append("<font color=\"#FF0000\">ERROR : 点云数据没有在模型中进行预测</font>")

        return
    
    if featfilepath == '':
        ui.textEdit.append("<font color=\"#FF0000\">ERROR : 未选择实际特征点文件夹</font>")

        return

    if ui.radioButton.isChecked():
        feature = 0
    elif ui.radioButton_2.isChecked():
        feature = 1
    elif ui.radioButton_3.isChecked():
        feature = 2
    elif ui.radioButton_4.isChecked():
        feature = 3
    else:
        feature = -1
    
    if feature == -1:
        return

    hwnd = win32gui.FindWindow(None , 'Point Cloud Visualization')
    if hwnd != 0: 
        global vis
        vis.destroy_window()
    vis = o3.visualization.Visualizer()

    filename = os.path.basename(filepath)
    name = filename.split(".")[0]
    filename = name + featurename[feature] + '.txt'
    filenameW = name + featurenameW[feature] + '.txt'
    filepath = './Visualization/pred_data/' + filename
    filepathW = './Visualization/pred_data/' + filenameW
    featlist = np.empty(shape = (0,3))
    for featfile in os.listdir(featfilepath):
        idx = featfile.split("_")[1]
        nameidx = name.split("_")[1]
        if idx == nameidx and featfile.endswith(feat[feature]):
            featpoint = np.loadtxt(featfilepath + '/' + featfile)
            featlist = np.vstack((featlist , featpoint))
    point_set = np.loadtxt(filepath)
    n = featlist.shape[0]
    rgb = np.zeros(shape = (n , 3))
    #实际特征点颜色
    for r in rgb:
        # r[0] = 1
        r[2] = 1
    featlist = np.hstack((featlist , rgb))
    cent_eh = duplif(n , featlist)
    point_set = np.vstack((point_set , featlist))
    point_set = np.vstack((point_set , cent_eh))
    np.savetxt(filepathW , point_set)

    source = o3.io.read_point_cloud(filename = str(filepathW),format = "xyzrgb")
    custom_draw_geometry(source)

def generateHeat(in_file, out_file):
    point_set = np.loadtxt(in_file)
    point_set = point_set[np.argsort(point_set[:,3])]

    featmax = point_set.max(axis=0)[3]
    featmin = point_set.min(axis=0)[3]

    # print(featmax)
    # print(featmin)

    for point in point_set:
        point[3] = (point[3] - featmin) / (featmax - featmin)
    point_set = point_set[:,[0,1,2,3]]
    n = point_set.shape[0]
    rgb = np.empty(shape = (n , 3))
    point_set = np.hstack((point_set , rgb))
    # for point in point_set:
    #     point[4] = 255 * point[3]
    #     point[5] = 0
    #     point[6] = 255 - 255 * point[3]
    for point in point_set:
         point[4] = point[3]
         point[5] = 0
         point[6] = 1 - point[3]
    point_set = np.delete(point_set, 3, axis=1)
    np.savetxt(out_file, point_set)

# 计算两个点之间的欧式距离，参数为两个元组
def dist(t1, t2):
    dis = math.sqrt(np.power((t1[0]-t2[0]),2) + np.power((t1[1]-t2[1]),2) + np.power((t1[2]-t2[2]),2))
    return dis
 
# DBSCAN算法，参数为数据集，Eps为指定半径参数，MinPts为制定邻域密度阈值
def dbscan(Data, Eps, MinPts):
    num = len(Data)  # 点的个数
    # print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]
    # C为输出结果，默认是一个长度为num的值全为-1的列表
    # 用k来标记不同的簇，k = -1表示噪声点
    k = -1
    # 如果还有没访问的点
    while len(unvisited) > 0:
        # 随机选择一个unvisited对象
        p = random.choice(unvisited)
        unvisited.remove(p)
        visited.append(p)
        # N为p的epsilon邻域中的对象的集合
        N = []
        for i in range(num):
            if (dist(Data[i], Data[p]) <= Eps):# and (i!=p):
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k+1
            # print(k)
            C[p] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if (dist(Data[j], Data[pi])<=Eps): #and (j!=pi):
                            M.append(j)
                    if len(M)>=MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1
 
    return C

# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist=[]
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids  #相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2     #平方
        squaredDist = np.sum(squaredDiff, axis=1)   #和  (axis=1表示行)
        distance = squaredDist ** 0.5  #开根号
        clalist.append(distance) 
    clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist

# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() #DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values
 
    # 计算变化量
    changed = newCentroids - centroids
 
    return changed, newCentroids

# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(list(dataSet), k)
    
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)
 
    centroids = sorted(newCentroids.tolist())   #tolist()将矩阵转换成列表 sorted()排序
 
    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k) #调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)  
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):   #enumerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])
        
    return centroids, cluster

# def calfeatCO(name):
#     point_set = np.loadtxt('Visualization/pred_data/'+ name + '.txt')
#     new_points = []
#     for p in point_set:
#         if p[3] > 0.37:
#             new_points.append(p)

#     c = dbscan(new_points, 1, 3)
#     idx = []
#     for i in range(len(c)):
#         if c[i] != -1:
#             idx.append(i)
#     c = np.array(c)
#     c = c[idx]
#     new_points = np.array(new_points)
#     new_points = new_points[idx,:]
#     new_points = new_points[:, 0:3]
#     n = len(c)
#     k = np.max(c)+1
#     rgb = np.zeros(shape = (n , 3))
#     for i in range(n):
#         seg = c[i]
#         rgb[i][0] = 1 - 1 * seg/k
#         rgb[i][1] = 0
#         rgb[i][2] = 0
#     new_points = np.hstack((new_points,rgb))
#     print(k)
#     np.savetxt('testdbscanCO.txt',new_points)

#     kpoints = [] #将点按聚类分成k类
#     for i in range(k): 
#         p = []
#         kpoints.append(p)
#     for i in range(n):
#         kpoints[c[i]].append(new_points[i][0:3])

#     cent = []
#     for kp in kpoints: #对每一类的全部点使用kmeans求一个聚类中心
#         centroids,_ = kmeans(kp, 1)
#         cent.append(centroids)
#     cent = np.array(cent)
#     cent = cent.reshape(k,-1)
    
#     rgb = np.zeros(shape = (k , 3))
#     for r in rgb:
#         r[1] = 1
#     cent = np.hstack((cent , rgb))
#     point_set = point_set[:,0:3]
#     N =point_set.shape[0]
#     rgb = np.zeros(shape = (N , 3))
#     for r in rgb:
#         r[0] = 0.5
#         r[1] = 0.5
#         r[2] = 0.5
#     point_set = np.hstack((point_set , rgb))
#     point_set = np.vstack((point_set,cent))
#     np.savetxt('testkmeanCO.txt',cent)
#     np.savetxt('Visualization/pred_data/' + name + '-PredCO.txt',point_set)

def dupli(ncent , cent_points):
    cent_eh = np.zeros(shape = (ncent * 24 , 6))
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + i*ncent][0] = cent_points[j][0]
            cent_eh[j + i*ncent][1] = cent_points[j][1]
            cent_eh[j + i*ncent][2] = cent_points[j][2]
            cent_eh[j + i*ncent][i] = cent_points[j][i] + 0.1
            cent_eh[j + i*ncent][5] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (3 + i)*ncent][0] = cent_points[j][0] 
            cent_eh[j + (3 + i)*ncent][1] = cent_points[j][1] 
            cent_eh[j + (3 + i)*ncent][2] = cent_points[j][2] 
            cent_eh[j + (3 + i)*ncent][i] = cent_points[j][i] - 0.1
            cent_eh[j + (3 + i)*ncent][5] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (6 + i)*ncent][0] = cent_points[j][0] + 0.1
            cent_eh[j + (6 + i)*ncent][1] = cent_points[j][1] + 0.1
            cent_eh[j + (6 + i)*ncent][2] = cent_points[j][2] + 0.1
            cent_eh[j + (6 + i)*ncent][i] = cent_points[j][i] - 0.1
            cent_eh[j + (6 + i)*ncent][5] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (9 + i)*ncent][0] = cent_points[j][0] - 0.1
            cent_eh[j + (9 + i)*ncent][1] = cent_points[j][1] - 0.1
            cent_eh[j + (9 + i)*ncent][2] = cent_points[j][2] - 0.1
            cent_eh[j + (9 + i)*ncent][i] = cent_points[j][i] + 0.1
            cent_eh[j + (9 + i)*ncent][5] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (12 + i)*ncent][0] = cent_points[j][0]
            cent_eh[j + (12 + i)*ncent][1] = cent_points[j][1]
            cent_eh[j + (12 + i)*ncent][2] = cent_points[j][2]
            cent_eh[j + (12 + i)*ncent][i] = cent_points[j][i] + 0.05
            cent_eh[j + (12 + i)*ncent][5] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (15 + i)*ncent][0] = cent_points[j][0] 
            cent_eh[j + (15 + i)*ncent][1] = cent_points[j][1] 
            cent_eh[j + (15 + i)*ncent][2] = cent_points[j][2] 
            cent_eh[j + (15 + i)*ncent][i] = cent_points[j][i] - 0.05
            cent_eh[j + (15 + i)*ncent][5] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (18 + i)*ncent][0] = cent_points[j][0] + 0.05
            cent_eh[j + (18 + i)*ncent][1] = cent_points[j][1] + 0.05
            cent_eh[j + (18 + i)*ncent][2] = cent_points[j][2] + 0.05
            cent_eh[j + (18 + i)*ncent][i] = cent_points[j][i] - 0.05
            cent_eh[j + (18 + i)*ncent][5] = 1
    for i in range(3):
        for j in range(len(cent_points)):
            cent_eh[j + (21 + i)*ncent][0] = cent_points[j][0] - 0.05
            cent_eh[j + (21 + i)*ncent][1] = cent_points[j][1] - 0.05
            cent_eh[j + (21 + i)*ncent][2] = cent_points[j][2] - 0.05
            cent_eh[j + (21 + i)*ncent][i] = cent_points[j][i] + 0.05
            cent_eh[j + (21 + i)*ncent][5] = 1
    return cent_eh

def calfeatCO(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-CO.txt')
    segdict = {}
    for p in point_set:
        if p[3] > 0.19:
            seg = int(p[4])
            if seg in segdict.keys():
                segdict[seg].append(p)
            else:
                segdict[seg] = []
                segdict[seg].append(p)

    cent_points = np.empty(shape = (0 , 6))
    for key in segdict.keys():
        v = segdict[key]
        new_points = np.array(v)
        new_points = new_points[:, 0:3]

        n = new_points.shape[0]
        rgb = np.zeros(shape = (n , 3))
        new_points = np.hstack((new_points,rgb))

        cent,_ = kmeans(new_points[:, 0:3], 2)
        cent = np.array(cent)
    
        rgb = np.zeros(shape = (cent.shape[0] , 3))
        for r in rgb:
            r[0] = 1
        cent = np.hstack((cent , rgb))
        cent_points = np.vstack((cent_points , cent))
        
    point_set = point_set[:,0:3]
    N =point_set.shape[0]
    rgb = np.zeros(shape = (N , 3))
    for r in rgb:
        r[0] = 0.8
        r[1] = 0.8
        r[2] = 0.8
    point_set = np.hstack((point_set , rgb))

    ncent = cent_points.shape[0]
    cent_eh = dupli(ncent , cent_points)
    point_set = np.vstack((point_set, cent_points))
    point_set = np.vstack((point_set, cent_eh))
    # np.savetxt('testkmeanCO.txt',cent_points)
    np.savetxt('Visualization/pred_data/' + name + '-PredCO.txt',point_set)

# def calfeatCU(name):
#     point_set = np.loadtxt('Visualization/pred_data/'+ name + '.txt')
#     new_points = []
#     for p in point_set:
#         if p[4] > 0.37:
#             new_points.append(p)

#     c = dbscan(new_points, 1, 3)
#     idx = []
#     for i in range(len(c)):
#         if c[i] != -1:
#             idx.append(i)
#     c = np.array(c)
#     c = c[idx]
#     new_points = np.array(new_points)
#     new_points = new_points[idx,:]
#     new_points = new_points[:, 0:3]
#     n = len(c)
#     k = np.max(c)+1
#     rgb = np.zeros(shape = (n , 3))
#     for i in range(n):
#         seg = c[i]
#         rgb[i][0] = 1 - 1 * seg/k
#         rgb[i][1] = 0
#         rgb[i][2] = 0
#     new_points = np.hstack((new_points,rgb))

#     print(k)
#     np.savetxt('testdbscanCU.txt',new_points)

#     kpoints = [] #将点按聚类分成k类
#     for i in range(k): 
#         p = []
#         kpoints.append(p)
#     for i in range(n):
#         kpoints[c[i]].append(new_points[i][0:3])

#     cent = []
#     for kp in kpoints: #对每一类的全部点使用kmeans求一个聚类中心
#         centroids,_ = kmeans(kp, 1)
#         cent.append(centroids)
#     cent = np.array(cent)
#     cent = cent.reshape(k,-1)
    
#     rgb = np.zeros(shape = (k , 3))
#     for r in rgb:
#         r[1] = 1
#     cent = np.hstack((cent , rgb))
#     point_set = point_set[:,0:3]
#     N =point_set.shape[0]
#     rgb = np.zeros(shape = (N , 3))
#     for r in rgb:
#         r[0] = 0.5
#         r[1] = 0.5
#         r[2] = 0.5
#     point_set = np.hstack((point_set , rgb))
#     point_set = np.vstack((point_set,cent))
#     np.savetxt('testkmeanCU.txt',cent)
#     np.savetxt('Visualization/pred_data/' + name + '-PredCU.txt',point_set)

def calfeatCU(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-CU.txt')
    segdict = {}
    for p in point_set:
        if p[3] > 0.25:
            seg = int(p[4])
            if seg in segdict.keys():
                segdict[seg].append(p)
            else:
                segdict[seg] = []
                segdict[seg].append(p)

    cent_points = np.empty(shape = (0 , 6))
    for key in segdict.keys():
        v = segdict[key]
        new_points = np.array(v)
        new_points = new_points[:, 0:3]

        n = new_points.shape[0]
        rgb = np.zeros(shape = (n , 3))
        new_points = np.hstack((new_points,rgb))

        if key in [11,12,21,22]:
            cent,_ = kmeans(new_points[:, 0:3], 3)
        if key in [13,23]:
            cent,_ = kmeans(new_points[:, 0:3], 1)
        if key in [14,15,24,25]:
            cent,_ = kmeans(new_points[:, 0:3], 2)
        if key in [16,17,26,27]:
            cent,_ = kmeans(new_points[:, 0:3], 4)
        cent = np.array(cent)
    
        rgb = np.zeros(shape = (cent.shape[0] , 3))
        for r in rgb:
            r[0] = 1
        cent = np.hstack((cent , rgb))
        cent_points = np.vstack((cent_points , cent))
        
    point_set = point_set[:,0:3]
    N =point_set.shape[0]
    rgb = np.zeros(shape = (N , 3))
    for r in rgb:
        r[0] = 0.8
        r[1] = 0.8
        r[2] = 0.8
    point_set = np.hstack((point_set , rgb))

    ncent = cent_points.shape[0]
    cent_eh = dupli(ncent , cent_points)
    point_set = np.vstack((point_set, cent_points))
    point_set = np.vstack((point_set, cent_eh))
    np.savetxt('Visualization/pred_data/' + name + '-PredCU.txt',point_set)

# def calfeatFA(name):
#     point_set = np.loadtxt('Visualization/pred_data/'+ name + '.txt')
#     new_points = []
#     for p in point_set:
#         if p[5] > 0.37:
#             new_points.append(p)

#     c = dbscan(new_points, 1, 3)
#     idx = []
#     for i in range(len(c)):
#         if c[i] != -1:
#             idx.append(i)
#     c = np.array(c)
#     c = c[idx]
#     new_points = np.array(new_points)
#     new_points = new_points[idx,:]
#     new_points = new_points[:, 0:3]
#     n = len(c)
#     k = np.max(c)+1
#     rgb = np.zeros(shape = (n , 3))
#     for i in range(n):
#         seg = c[i]
#         rgb[i][0] = 1 - 1 * seg/k
#         rgb[i][1] = 0
#         rgb[i][2] = 0
#     new_points = np.hstack((new_points,rgb))

#     print(k)
#     np.savetxt('testdbscanFA.txt',new_points)

#     kpoints = [] #将点按聚类分成k类
#     for i in range(k): 
#         p = []
#         kpoints.append(p)
#     for i in range(n):
#         kpoints[c[i]].append(new_points[i][0:3])

#     cent = []
#     for kp in kpoints: #对每一类的全部点使用kmeans求一个聚类中心
#         centroids,_ = kmeans(kp, 1)
#         cent.append(centroids)
#     cent = np.array(cent)
#     cent = cent.reshape(k,-1)
    
#     rgb = np.zeros(shape = (k , 3))
#     for r in rgb:
#         r[1] = 1
#     cent = np.hstack((cent , rgb))
#     point_set = point_set[:,0:3]
#     N =point_set.shape[0]
#     rgb = np.zeros(shape = (N , 3))
#     for r in rgb:
#         r[0] = 0.5
#         r[1] = 0.5
#         r[2] = 0.5
#     point_set = np.hstack((point_set , rgb))
#     point_set = np.vstack((point_set,cent))
#     np.savetxt('testkmeanFA.txt',cent)
#     np.savetxt('Visualization/pred_data/' + name + '-PredFA.txt',point_set)

def calfeatFA(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-FA.txt')
    segdict = {}
    for p in point_set:
        if p[3] > 0.155:
            seg = int(p[4])
            if seg in segdict.keys():
                segdict[seg].append(p)
            else:
                segdict[seg] = []
                segdict[seg].append(p)

    cent_points = np.empty(shape = (0 , 6))
    for key in segdict.keys():
        v = segdict[key]
        new_points = np.array(v)
        new_points = new_points[:, 0:3]

        n = new_points.shape[0]
        rgb = np.zeros(shape = (n , 3))
        new_points = np.hstack((new_points,rgb))

        cent,_ = kmeans(new_points[:, 0:3], 1)
        cent = np.array(cent)
    
        rgb = np.zeros(shape = (cent.shape[0] , 3))
        for r in rgb:
            r[0] = 1
        cent = np.hstack((cent , rgb))
        cent_points = np.vstack((cent_points , cent))
        
    point_set = point_set[:,0:3]
    N =point_set.shape[0]
    rgb = np.zeros(shape = (N , 3))
    for r in rgb:
        r[0] = 0.8
        r[1] = 0.8
        r[2] = 0.8
    point_set = np.hstack((point_set , rgb))

    ncent = cent_points.shape[0]
    cent_eh = dupli(ncent , cent_points)
    point_set = np.vstack((point_set, cent_points))
    point_set = np.vstack((point_set, cent_eh))
    point_set = np.vstack((point_set, cent_points))

    np.savetxt('Visualization/pred_data/' + name + '-PredFA.txt',point_set)

# def calfeatOC(name):
#     point_set = np.loadtxt('Visualization/pred_data/'+ name + '.txt')
#     new_points = []
#     for p in point_set:
#         if p[6] > 0.37:
#             new_points.append(p)

#     c = dbscan(new_points, 1, 3)
#     idx = []
#     for i in range(len(c)):
#         if c[i] != -1:
#             idx.append(i)
#     c = np.array(c)
#     c = c[idx]
#     new_points = np.array(new_points)
#     new_points = new_points[idx,:]
#     new_points = new_points[:, 0:3]
#     n = len(c)
#     k = np.max(c)+1
#     rgb = np.zeros(shape = (n , 3))
#     for i in range(n):
#         seg = c[i]
#         rgb[i][0] = 1 - 1 * seg/k
#         rgb[i][1] = 0
#         rgb[i][2] = 0
#     new_points = np.hstack((new_points,rgb))

#     print(k)
#     np.savetxt('testdbscanOC.txt',new_points)
#     # cent = np.zeros(shape = (14 , 6))
#     # for i in range(n):
#     #     cent[c[i]] += new_points[i]
#     # cent = cent/14
#     # for p in cent:
#     #     p[3] = 255
#     # new_points = np.vstack((new_points, cent))
#     # np.savetxt('testdbscanMean.txt',new_points)

#     kpoints = [] #将点按聚类分成k类
#     for i in range(k): 
#         p = []
#         kpoints.append(p)
#     for i in range(n):
#         kpoints[c[i]].append(new_points[i][0:3])

#     cent = []
#     for kp in kpoints: #对每一类的全部点使用kmeans求一个聚类中心
#         centroids,_ = kmeans(kp, 1)
#         cent.append(centroids)
#     cent = np.array(cent)
#     cent = cent.reshape(k,-1)
    
#     rgb = np.zeros(shape = (k , 3))
#     for r in rgb:
#         r[1] = 1
#     cent = np.hstack((cent , rgb))
#     point_set = point_set[:,0:3]
#     N =point_set.shape[0]
#     rgb = np.zeros(shape = (N , 3))
#     for r in rgb:
#         r[0] = 0.5
#         r[1] = 0.5
#         r[2] = 0.5
#     point_set = np.hstack((point_set , rgb))
#     point_set = np.vstack((point_set,cent))
#     np.savetxt('testkmeanOC.txt',cent)
#     np.savetxt('Visualization/pred_data/' + name + '-PredOC.txt',point_set)

def calfeatOC(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-OC.txt')
    segdict = {}
    for p in point_set:
        if p[3] > 0.23:
            seg = int(p[4])
            if seg in segdict.keys():
                segdict[seg].append(p)
            else:
                segdict[seg] = []
                segdict[seg].append(p)

    cent_points = np.empty(shape = (0 , 6))
    for key in segdict.keys():
        v = segdict[key]
        new_points = np.array(v)
        new_points = new_points[:, 0:3]

        n = new_points.shape[0]
        rgb = np.zeros(shape = (n , 3))
        new_points = np.hstack((new_points,rgb))

        cent,_ = kmeans(new_points[:, 0:3], 2)
        cent = np.array(cent)
    
        rgb = np.zeros(shape = (cent.shape[0] , 3))
        for r in rgb:
            r[0] = 1
        cent = np.hstack((cent , rgb))
        cent_points = np.vstack((cent_points , cent))
        
    point_set = point_set[:,0:3]
    N =point_set.shape[0]
    rgb = np.zeros(shape = (N , 3))
    for r in rgb:
        r[0] = 0.8
        r[1] = 0.8
        r[2] = 0.8
    point_set = np.hstack((point_set , rgb))

    ncent = cent_points.shape[0]
    cent_eh = dupli(ncent , cent_points)
    point_set = np.vstack((point_set, cent_points))
    point_set = np.vstack((point_set, cent_eh))
    point_set = np.vstack((point_set, cent_points))
    np.savetxt('Visualization/pred_data/' + name + '-PredOC.txt',point_set)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc,m,centroid

def vert2matrix(inputVert):
    '''
    transform labels from vert to matrix
    '''
    N = len(inputVert)
    outputMat = np.zeros(shape = (1, N, 14))
    # print(outputMat)
    for i in range(N):
        seg = inputVert[i]
        m = 19 - int(seg)
        if m > 0:
            p = 8 - m
            outputMat[0,i,p] = 1
        else:
            m = -1 * m
            p = m + 5
            outputMat[0,i,p] = 1
    return outputMat


load_mutex = 0
pre_mutex = 0
heat_mutex = 0
feat_mutex = 0

class myThread (threading.Thread):
    def __init__(self, fileName_choose,name,point_set):
        threading.Thread.__init__(self)
        self.fileName_choose = fileName_choose
        self.name = name
        self.point_set = point_set
    def run(self):
        LoadThread(self.fileName_choose, self.name, self.point_set)
 


def LoadThread(fileName_choose, name, point_set):
    # 加载模型
    '''MODEL LOADING'''

    MODEL = importlib.import_module('all_teeth_ver11')
    # classifier = MODEL.get_model().cuda()
    classifierCO = MODEL.get_model(14)
    checkpointCO = torch.load('log/latest_modelCO.pth')
    classifierCO.load_state_dict(checkpointCO['model_state_dict'])

    classifierCU = MODEL.get_model(14)
    checkpointCU = torch.load('log/latest_modelCU.pth')
    classifierCU.load_state_dict(checkpointCU['model_state_dict'])

    classifierFA = MODEL.get_model(14)
    checkpointFA = torch.load('log/latest_modelFA.pth')
    classifierFA.load_state_dict(checkpointFA['model_state_dict'])

    classifierOC = MODEL.get_model(14)
    checkpointOC = torch.load('log/latest_modelOC.pth')
    classifierOC.load_state_dict(checkpointOC['model_state_dict'])
    
    global load_mutex    
    load_mutex = 1
   

    with torch.no_grad():

        points = point_set[:, 0:3]
        point_label = point_set[:,7]
        points,m,centroid = pc_normalize(points) 

        points = points.reshape(1,-1,3)
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        # points = points.cuda()

        seg = vert2matrix(point_label)
        seg = torch.Tensor(seg)
        seg = seg.transpose(2, 1)
        # seg = seg.cuda()


        pred_CO = classifierCO(points , seg)
        points = points.transpose(2, 1)
        pred_CO = pred_CO.transpose(2, 1)
        points = points.reshape(-1, 3)
        pred_CO = pred_CO.reshape(-1, 1)
        point_label = point_label.reshape(-1, 1)

        xyz = points.numpy() * m + centroid
        pred_CO = pred_CO.numpy()

        savepath = 'Visualization/pred_data/'
        
        np.savetxt(savepath + name + '-CO.txt',np.concatenate((xyz,pred_CO,point_label),axis=1))

    with torch.no_grad():

        points = point_set[:, 0:3]
        point_label = point_set[:,7]
        points,m,centroid = pc_normalize(points) 

        points = points.reshape(1,-1,3)
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        # points = points.cuda()

        seg = vert2matrix(point_label)
        seg = torch.Tensor(seg)
        seg = seg.transpose(2, 1)
        # seg = seg.cuda()


        pred_CU = classifierCU(points , seg)
        points = points.transpose(2, 1)
        pred_CU = pred_CU.transpose(2, 1)
        points = points.reshape(-1, 3)
        pred_CU = pred_CU.reshape(-1, 1)
        point_label = point_label.reshape(-1, 1)

        xyz = points.numpy() * m + centroid
        pred_CU = pred_CU.numpy()

        savepath = 'Visualization/pred_data/'
        
        np.savetxt(savepath + name + '-CU.txt',np.concatenate((xyz,pred_CU,point_label),axis=1))
    
    with torch.no_grad():

        points = point_set[:, 0:3]
        point_label = point_set[:,7]
        points,m,centroid = pc_normalize(points) 

        points = points.reshape(1,-1,3)
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        # points = points.cuda()

        seg = vert2matrix(point_label)
        seg = torch.Tensor(seg)
        seg = seg.transpose(2, 1)
        # seg = seg.cuda()


        pred_FA = classifierFA(points , seg)
        points = points.transpose(2, 1)
        pred_FA = pred_FA.transpose(2, 1)
        points = points.reshape(-1, 3)
        pred_FA = pred_FA.reshape(-1, 1)
        point_label = point_label.reshape(-1, 1)

        xyz = points.numpy() * m + centroid
        pred_FA = pred_FA.numpy()

        savepath = 'Visualization/pred_data/'
        
        np.savetxt(savepath + name + '-FA.txt',np.concatenate((xyz,pred_FA,point_label),axis=1))

    with torch.no_grad():

        points = point_set[:, 0:3]
        point_label = point_set[:,7]
        points,m,centroid = pc_normalize(points) 

        points = points.reshape(1,-1,3)
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        # points = points.cuda()

        seg = vert2matrix(point_label)
        seg = torch.Tensor(seg)
        seg = seg.transpose(2, 1)
        # seg = seg.cuda()


        pred_OC = classifierOC(points , seg)
        points = points.transpose(2, 1)
        pred_OC = pred_OC.transpose(2, 1)
        points = points.reshape(-1, 3)
        pred_OC = pred_OC.reshape(-1, 1)
        point_label = point_label.reshape(-1, 1)

        xyz = points.numpy() * m + centroid
        pred_OC = pred_OC.numpy()

        savepath = 'Visualization/pred_data/'
        
        np.savetxt(savepath + name + '-OC.txt',np.concatenate((xyz,pred_OC,point_label),axis=1))



    global pre_mutex   
    pre_mutex = 1

    generateHeat('Visualization/pred_data/'+ name + '-CO.txt', 'Visualization/pred_data/' + name + '-heatCO.txt')
    generateHeat('Visualization/pred_data/'+ name + '-CU.txt', 'Visualization/pred_data/' + name + '-heatCU.txt')
    generateHeat('Visualization/pred_data/'+ name + '-FA.txt', 'Visualization/pred_data/' + name + '-heatFA.txt')
    generateHeat('Visualization/pred_data/'+ name + '-OC.txt', 'Visualization/pred_data/' + name + '-heatOC.txt')

    global heat_mutex    
    heat_mutex = 1

    
    #聚类算法生成特征点
    # to do
    calfeatCO(name)
    calfeatCU(name)
    calfeatFA(name)
    calfeatOC(name)
    global feat_mutex    
    feat_mutex = 1


# class Worker(QThread):
#     sinOut = Signal(str, int)

#     def __init__(self, parent=None, point_set=None, name=None):
#         super(Worker, self).__init__(parent)
#         #设置工作状态与初始num数值
#         self.working = True
#         self.num = 0
#         self.point_set = point_set
#         self.name = name

#     def __del__(self):
#         #线程状态改变与线程终止
#         self.working = False
#         self.wait()

#     def run(self):
#         point_set = self.point_set
#         name = self.name
#         log_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Loading Model'
#         self.sinOut.emit(log_str,20)
            
#         MODEL = importlib.import_module('all_teeth_ver01')
#         # classifier = MODEL.get_model().cuda()
#         classifier = MODEL.get_model()
#         checkpoint = torch.load('log/best_model2.pth')   


#         log_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Predicting'
#         self.sinOut.emit(log_str,50)

#         with torch.no_grad():

#             points = point_set[:, 0:3]
#             points,m,centroid = pc_normalize(points) 

#             points = points.reshape(1,-1,3)
#             points = torch.Tensor(points)
#             points = points.transpose(2, 1)
#             # points = points.cuda()


#             pred_CO = classifier(points)
#             points = points.transpose(2, 1)
#             pred_CO = pred_CO.transpose(2, 1)
#             points = points.reshape(-1,3)
#             pred_CO = pred_CO.reshape(-1,1)

#             xyz = points.numpy() * m + centroid
#             pred_CO = pred_CO.numpy()

#             savepath = 'Visualization/pred_data/'
                
#             np.savetxt(savepath + name + '.txt',np.concatenate((xyz,pred_CO),axis=1))

#         log_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Generating HeatMaps'
#         self.sinOut.emit(log_str,70)
#         generateHeat(0, 'Visualization/pred_data/'+ name + '.txt', 'Visualization/pred_data/' + name + '-heatCO.txt')
#         # generateHeat(1, "", './Visualization/pred_data/' + 'model name' + '-CU.txt')
#         # generateHeat(2, "", './Visualization/pred_data/' + 'model name' + '-FA.txt')
#         # generateHeat(3, "", './Visualization/pred_data/' + 'model name' + '-OC.txt')
#         log_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Calculating Feature Points'
#         self.sinOut.emit(log_str,90)
#         #聚类算法生成特征点
#         # to do
            
#         log_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Success!!!'
#         self.sinOut.emit(log_str,100)
            

    
# def slotUp(log_str, value):
#     ui.textEdit.append(log_str)
#     ui.progressBar.setValue(value)

pv = 0

mutex = 0

def checkload():
    global load_mutex
    global pv
    if pv <= 30:   
        pv = pv + 1
    ui.progressBar.setValue(pv)
    if load_mutex == 1:      
        ui.timer1.stop()
        ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Predicting')
        global mutex
        mutex += 1

def checkpre():
    global load_mutex
    if load_mutex == 1:
        global pre_mutex
        global pv
        if pv <= 80:   
            pv = pv + 1
        ui.progressBar.setValue(pv)
        if pre_mutex == 1:      
            ui.timer2.stop()
            ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Generating HeatMaps')
            global mutex
            mutex += 1

def checkheat():
    global pre_mutex
    if pre_mutex == 1:
        global heat_mutex
        global pv
        pv = pv + 1
        ui.progressBar.setValue(pv)
        if heat_mutex == 1:      
            ui.timer3.stop()
            ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Calculating Feature Points')
            global mutex
            mutex += 1

def checkfeat():
    global heat_mutex
    if heat_mutex == 1:
        global feat_mutex
        global pv
        pv = pv + 1
        ui.progressBar.setValue(pv)
        if feat_mutex == 1:      
            global load_mutex 
            global pre_mutex 
            if load_mutex == 1 and pre_mutex == 1 and heat_mutex == 1 and feat_mutex == 1:
                global mutex
                if mutex == 3:
                    load_mutex = 0
                    pre_mutex = 0
                    heat_mutex = 0
                    feat_mutex = 0
                    mutex = 0
                    pv = 0
                    ui.timer4.stop()
                    ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Success!!!')
                    ui.progressBar.setValue(100)


def getTeethFile(self):
    '''
    读取原始点云文件进行预测
    '''

    #读取文件名
    fileName_choose, _ = QFileDialog.getOpenFileName(None, "选取牙冠文件", os.getcwd(),"All Files (*)")
    if fileName_choose == "":
        return
    ui.lineEdit.setText(fileName_choose)
    name = os.path.basename(fileName_choose)
    name = name.split(".")[0]

    ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Loading Point Cloud File')

    #加载点云
    point_set = np.loadtxt(fileName_choose)
    ui.progressBar.setValue(0)

    # ui.thread = Worker(None, point_set, name)
    # ui.thread.sinOut.connect(slotUp)
    # ui.thread.start()

    thread1 = myThread(fileName_choose,name,point_set)
    thread1.start()


    ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Loading Model')

    ui.timer1.start(300)
    
    ui.timer2.start(500)
    
    ui.timer3.start(100)
    
    ui.timer4.start(100)

    global TRAINED
    TRAINED = True




    

    # ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Loading Model')
    # #加载模型
    # '''MODEL LOADING'''
    # MODEL = importlib.import_module('all_teeth_ver01')
    # global classifier
    # # classifier = MODEL.get_model().cuda()
    # classifier = MODEL.get_model()
    # checkpoint = torch.load('log/best_model2.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])

    # ui.progressBar.setValue(40)

    # ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Predicting')

    # with torch.no_grad():

    #     points = point_set[:, 0:3]
    #     points,m,centroid = pc_normalize(points) 

    #     points = points.reshape(1,-1,3)
    #     points = torch.Tensor(points)
    #     points = points.transpose(2, 1)
    #     # points = points.cuda()

    #     pred_CO = classifier(points)
    #     points = points.transpose(2, 1)
    #     pred_CO = pred_CO.transpose(2, 1)
    #     points = points.reshape(-1,3)
    #     pred_CO = pred_CO.reshape(-1,1)

    #     xyz = points.numpy() * m + centroid
    #     pred_CO = pred_CO.numpy()

    #     savepath = 'Visualization/pred_data/'
        
    #     np.savetxt(savepath + name + '.txt',np.concatenate((xyz,pred_CO),axis=1))
    
    # ui.progressBar.setValue(60)

    # ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Generating HeatMaps')

    # # generate 4 heatmap
    # # to do
    # generateHeat(0, 'Visualization/pred_data/'+ name + '.txt', 'Visualization/pred_data/' + name + '-heatCO.txt')
    # ui.progressBar.setValue(65)
    # # generateHeat(1, "", './Visualization/pred_data/' + 'model name' + '-CU.txt')
    # ui.progressBar.setValue(70)
    # # generateHeat(2, "", './Visualization/pred_data/' + 'model name' + '-FA.txt')
    # ui.progressBar.setValue(75)
    # # generateHeat(3, "", './Visualization/pred_data/' + 'model name' + '-OC.txt')
    # ui.progressBar.setValue(80)

    # ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Calculating Feature Points')
    # #聚类算法生成特征点
    # # to do
    # ui.progressBar.setValue(100)

    # ui.textEdit.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' : Success!!!')
    
    

def getFeatureFile(self):
    fileName_choose = QFileDialog.getExistingDirectory(None, "选取特征文件", os.getcwd()) 
    if fileName_choose == "":
        return
    ui.lineEdit_1.setText(fileName_choose)

def closeVisualize(self):
    global vis
    if vis == None:
        return
    vis.destroy_window()

class MainWindow(QtWidgets.QMainWindow):
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self,
                                               '提示',
                                               "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            closeVisualize(self)
            global p
            p.terminate()
            event.accept()
        else:
            event.ignore()

class Puttopmost(Process):
    def __init__(self, p):
        super().__init__()
        self.p = p
    def run(self):
        while True:
            hwnd = win32gui.FindWindow(None , 'Point Cloud Visualization')
            if hwnd == 0:
                continue
            else:
                # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 600, 500, win32con.SWP_NOSIZE|win32con.SWP_NOMOVE) 
                win32gui.SetParent(hwnd, self.p)
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                while win32gui.FindWindow(None , 'Point Cloud Visualization') != 0:
                    continue

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.pushButton.clicked.connect(lambda: open_file_xyz(ui.lineEdit.text()))
    ui.pushButton_2.clicked.connect(lambda: open_file_heat(ui.lineEdit.text()))
    ui.pushButton_3.clicked.connect(lambda: draw_pre_feature(ui.lineEdit.text()))
    ui.pushButton_4.clicked.connect(lambda: draw_feature(ui.lineEdit.text() , ui.lineEdit_1.text()))
    # ui.pushButton_3.clicked.connect(click3)
    ui.toolButton.clicked.connect(getTeethFile)
    ui.toolButton_1.clicked.connect(getFeatureFile)
    MainWindow.destroyed.connect(closeVisualize)

    p = Puttopmost(int(ui.widget.winId()))
    p.daemon = True
    p.start()

    MainWindow.show()
    sys.exit(app.exec_())