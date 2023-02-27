import matplotlib.pyplot as plt
import numpy as np
import PyQt5.Qt as Qt
import PyQt5.QtCore as QtCore
import sys

from collections import namedtuple
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib import lines
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout

from qpsolvers import solve_qp


FPS = 20                # simulation frames per second
X_MIN = -np.pi          # simulation environment size along x [m]
X_MAX = 3*np.pi         # simulation environment size along x [m]

m = 1.0                 # car mass [kg]
a = 0.5                 # hill scaling [m]
u_min = -4.0            # min force [N]
u_max = 4.0             # max force [N]

k1 = 0.6                # dynamic friction [1/s] (only for simulator)
k2 = 0.3                # drag [1/m] (only for simulator)
g = 9.81                # acceleration due to gravity (DO NOT CHANGE -- unless you change planet...)
sigma_x = 0.00          # simulated noise magnitude on position [m]
sigma_v = 0.1          # simulated noise magnitude on velocity [m/s]

class Controller:
    def __init__(self):
        self.kp = 100.0
    def control(self, x: float, v: float, r_x:float):
        u = self.kp*(r_x-x) # TODO: change this simple P-controller to LQR/MPC...!
        return u

class CarSimulator:

    def __init__(self, x: float, v: float):
        self.x = x
        self.v = v
        self.r_x = np.pi/2.0

    def update(self, dt: float, u: float):
        
        # kinematics using Euler-forward discretisation
        x_dot = self.v/np.sqrt((a*np.cos(self.x))**2 + 1.0)
        v_dot = -a*g*np.cos(self.x)/np.sqrt((a*np.cos(self.x))**2 + 1.0) - (k1*self.v + k2*self.v*np.abs(k1*self.v)) + u/m
        
        self.x = self.x + dt * x_dot + np.random.normal(0, sigma_x)
        self.v = self.v + dt * v_dot + np.random.normal(0, sigma_v)

class Window(QWidget):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setMouseTracking(True)

        self.carSimulator = CarSimulator(x=0.5*np.pi, v=0)
        #self.controller = LqrController()
        #self.controller = MpcController()
        self.controller = Controller()
        self.k = 0

        # set the layout
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.cid = self.canvas.mpl_connect('button_press_event', self)
        
        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        screen_size = self.screen().geometry().size()
        min_size = min(screen_size.width(), screen_size.height())
        self.resize(QtCore.QSize(min_size / 2, min_size / 2))

        # Timer for updating the view, with a delta t of 1s / fps between frames.
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000.0 / FPS)  # in milliseconds
        
    def __call__(self, event):
        modifiers = QApplication.keyboardModifiers()
        Mmodo = QApplication.mouseButtons()
        if Mmodo == QtCore.Qt.LeftButton:
            print('reset car x =',event.xdata)
            self.carSimulator.x = event.xdata
            self.carSimulator.v = 0.0
        else:
            print('reset reference position x =',event.xdata)
            self.carSimulator.r_x = event.xdata

    def update(self):
        # call controller
        u=self.controller.control(self.carSimulator.x, self.carSimulator.v, self.carSimulator.r_x)
        # saturate
        u = max(u_min,u)
        u = min(u_max,u)
        # simulate all together
        self.carSimulator.update(dt=1 / FPS, u=u)
        self.plot()
        self.k += 1

    ####################################################################################################################
    # Rendering ########################################################################################################
    ####################################################################################################################
    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(-a*1.5, a*1.5)

        x = self.carSimulator.x
        h = a*np.sin(x)
        
        # draw hills
        x_line = np.arange(X_MIN, X_MAX, 0.01)   # start,stop,step
        h_line = a*np.sin(x_line)
        plt.plot(x_line, h_line)
        
        # reference
        plt.plot(self.carSimulator.r_x, a*np.sin(self.carSimulator.r_x), '|g', markersize=12)
        
        # car as a blob
        plt.plot(x, h, 'or')
        
        # explanation
        plt.text(X_MIN+0.1, a*1.3, 'left click: reset ball position x')
        plt.text(X_MIN+0.1, -a*1.3, 'right click: reset ball reference r_x')
        
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
