import dataclasses
import sys

from typing import List
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtGui import QPainter, QBrush, QPen, QFont
from PyQt5.QtCore import Qt, QRect

import numpy as np
import math

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


X_SIZE = 600
X_CELLS = 10
Y_CELLS = 10
GAMMA = 0.9
LIVING_REWARD = -0.1
P_CORRECT = 0.8
DRAW_V = True

CELL_SIZE = int(round(float(X_SIZE)/float(X_CELLS)))
Y_SIZE = int(round(Y_CELLS*CELL_SIZE))

class MDP:
    def __init__(self):
        self.V = np.zeros((X_CELLS, Y_CELLS))
        self.Q = np.zeros((X_CELLS, Y_CELLS, 4))
    
    def solve(self, env, num_iter):
        print("TODO: solve me!")

class Environment2D:
    def __init__(self) -> None:
        self.obstacle = np.zeros((X_CELLS, Y_CELLS))
        self.fire = np.zeros((X_CELLS, Y_CELLS))
        self.cash = np.zeros((X_CELLS, Y_CELLS))
        self.mdp = MDP()

    def plan(self, num_iter):
        self.mdp.solve(self, num_iter)

    def add_obstacle(self, i, j):
        if self.obstacle[i,j] == 0 and self.fire[i,j] == 0 and self.cash[i,j] == 0:
            self.obstacle[i,j] = 1
        else:
            self.obstacle[i,j] = 0
        
    def add_fire(self, i, j):
        if self.obstacle[i,j] == 0 and self.fire[i,j] == 0 and self.cash[i,j] == 0:
            self.fire[i,j] = 1
        else:
            self.fire[i,j] = 0
    
    def add_cash(self, i, j):
        if self.obstacle[i,j] == 0 and self.fire[i,j] == 0 and self.cash[i,j] == 0:
            self.cash[i,j] = 1
        else:
            self.cash[i,j] = 0

    def clear(self):
        self.obstacle = np.zeros((X_CELLS, Y_CELLS))
        self.fire = np.zeros((X_CELLS, Y_CELLS))
        self.cash = np.zeros((X_CELLS, Y_CELLS))
        

class PlanningVisualizationWidget(QtWidgets.QWidget):
    
    class DRAWING_STATE:
        DRAW_FIRE = 0
        DRAW_CASH = 1
        DRAW_OBSTACLE = 2

    def __init__(self, environment: Environment2D):
        super(PlanningVisualizationWidget, self).__init__()

        self.env = environment
        self.cache_box_start = None
        self.cache_box_end = None
        self.draw_status = self.DRAWING_STATE.DRAW_OBSTACLE

        width = X_SIZE
        height = Y_SIZE+CELL_SIZE
        self.setGeometry(30, 30, width, height)
        self.setFixedSize(width, height)
        self.setWindowTitle("MDP Visualization Tool")
        
        # a figure instance to plot on
        #self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        #self.canvas = FigureCanvas(self.figure)

        self.area = QtWidgets.QWidget(self)
        self.obstacle_button = QtWidgets.QPushButton("obstacle")
        self.obstacle_button.setStyleSheet('QPushButton {color: gray;}')
        self.obstacle_button.clicked.connect(self.set_draw_obstacle)
        self.fire_button = QtWidgets.QPushButton("fire")
        self.fire_button.setStyleSheet('QPushButton {color: red;}')
        self.fire_button.clicked.connect(self.set_draw_fire)
        self.cash_button = QtWidgets.QPushButton("cash")
        self.cash_button.setStyleSheet('QPushButton {color: green;}')
        self.cash_button.clicked.connect(self.set_draw_cash)

        self.plan_button = QtWidgets.QPushButton("plan")
        self.plan_button.setStyleSheet('QPushButton {color: black;}')
        self.plan_button.clicked.connect(self.planning_callback)
        self.clear_button = QtWidgets.QPushButton("clear")
        self.clear_button.setStyleSheet('QPushButton {color: black;}')
        self.clear_button.clicked.connect(self.clear_environment)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.area, 0, 0)
        layout.addWidget(self.obstacle_button, 1, 0)
        layout.addWidget(self.fire_button, 1, 1)
        layout.addWidget(self.cash_button, 1, 2)
        layout.addWidget(self.plan_button, 1, 3)
        layout.addWidget(self.clear_button, 1, 4)

        # create an axis
        #self.axes = self.figure.add_subplot(111)

        # refresh canvas
        #self.canvas.draw()
        #self.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        self.setLayout(layout)
        self.show()

    ###############################################################################################
    # Button and Mouse Callbacks ##################################################################
    ###############################################################################################
    def set_draw_obstacle(self):
        self.draw_status = self.DRAWING_STATE.DRAW_OBSTACLE
        
    def set_draw_fire(self):
        self.draw_status = self.DRAWING_STATE.DRAW_FIRE
        
    def set_draw_cash(self):
        self.draw_status = self.DRAWING_STATE.DRAW_CASH
    
    def mousePressEvent(self, event):
        # All pressable points are in the borders in the environment by construction.
        if self.draw_status == self.DRAWING_STATE.DRAW_OBSTACLE:
            self.env.add_obstacle(int(math.floor(event.pos().x()/CELL_SIZE)), min(Y_CELLS-1, int(math.floor(event.pos().y()/CELL_SIZE))))

        elif self.draw_status == self.DRAWING_STATE.DRAW_FIRE:
            self.env.add_fire(int(math.floor(event.pos().x()/CELL_SIZE)), min(Y_CELLS-1, int(math.floor(event.pos().y()/CELL_SIZE))))

        elif self.draw_status == self.DRAWING_STATE.DRAW_CASH:
            self.env.add_cash(int(math.floor(event.pos().x()/CELL_SIZE)), min(Y_CELLS-1, int(math.floor(event.pos().y()/CELL_SIZE))))

        self.update()
    

    def clear_environment(self):
        self.env.clear()
        self.update()

    def planning_callback(self):
        self.env.plan(1000)
        self.update()

    ###############################################################################################
    # Painting ####################################################################################
    ###############################################################################################
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.gray, 1, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        for i in range(0,X_CELLS):
            for j in range(0, Y_CELLS):
                if self.env.fire[i,j]:
                    painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                elif self.env.cash[i,j]:
                    painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
                elif self.env.obstacle[i,j]:
                    painter.setBrush(QBrush(Qt.gray, Qt.SolidPattern))
                else:
                    painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                painter.setPen(QPen(Qt.gray, 1, Qt.SolidLine))
                rect = QRect(i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                painter.drawRect(rect)
                painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                if DRAW_V:
                    painter.setFont(QFont("Arial", int(CELL_SIZE/3)));
                    V_string = "{:.2f}".format(self.env.mdp.V[i,j])
                    painter.drawText(rect, Qt.AlignCenter, V_string)
                else:
                    painter.setFont(QFont("Arial", int(CELL_SIZE/6)));
                    Q_left = "{:.2f}".format(self.env.mdp.Q[i,j,0])
                    Q_right = "{:.2f}".format(self.env.mdp.Q[i,j,1])
                    Q_top = "{:.2f}".format(self.env.mdp.Q[i,j,2])
                    Q_bottom = "{:.2f}".format(self.env.mdp.Q[i,j,3])
                    painter.drawText(rect, Qt.AlignVCenter | Qt.AlignLeft, Q_left)
                    painter.drawText(rect, Qt.AlignVCenter | Qt.AlignRight, Q_right)
                    painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, Q_top)
                    painter.drawText(rect, Qt.AlignHCenter | Qt.AlignBottom, Q_bottom)
            
                            
if __name__ == '__main__':
    env = Environment2D()
    app = QtWidgets.QApplication(sys.argv)
    window = PlanningVisualizationWidget(env)
    window.show()
    sys.exit(app.exec_())
