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
u_min = -5.0            # min force [N]
u_max = 5.0             # max force [N]

k1 = 0.6                # dynamic friction [1/s] (only for simulator)
k2 = 0.3                # drag [1/m] (only for simulator)
g = 9.81                # acceleration due to gravity (DO NOT CHANGE -- unless you change planet...)
sigma_x = 0.00          # simulated noise magnitude on position [m]
sigma_v = 0.1          # simulated noise magnitude on velocity [m/s]

class LqrController:
    def __init__(self):
        self.Q_1 = 100.0
        self.Q_2 = 0.1
        self.R = 1
    def control(self, x: float, v: float, r_x:float):
        # note: we are just going to stabilise this on the first hill at pi/2, so the reference is ignored.
        P_12 = self.R*m*np.sqrt(a**2*g**2*m**2 + self.Q_1/self.R) + self.R*a*g*m**2 # negative solution leads to infeasible P_2
        P_2 = m*np.sqrt(self.R*(2*P_12 + self.Q_2))
        u = -1.0/(self.R*m)*(P_12*(x-np.pi/2.0) + P_2*v) # just wrote the matrix multiplication out...
        return u
        
class MpcController:
    def __init__(self):
        self.P_1 = 100.0
        self.P_2 = 0.1
        self.Q_1 = 100.0
        self.Q_2 = 0.1
        self.R = 0.001
        self.N = 100 # horizon
    def control(self, x: float, v: float, r_x:float):
        N = self.N
        dt = 1.0/FPS
        # first, we compute the linearised, discrete-time system:
        r_v = 0.0
        x_bar = x
        v_bar = 0.0
        F_c = np.array([[ (a**2*v_bar*np.cos(x_bar)*np.sin(x_bar))/(a**2*np.cos(x_bar)**2 + 1)**1.5, 1/np.sqrt(a**2*np.cos(x_bar)**2 + 1)],
                        [ (a*g*np.sin(x_bar))/(- a**2*np.sin(x_bar)**2 + a**2 + 1)**1.5, 0.0]]) # note the first entry is 0.0 with v_bar=0.0...
        G_c = np.array([[0],[1.0/m]])
        F = np.identity(2) + dt * F_c
        G = dt * G_c
        
        # linearisation point and deltas:
        u_bar = a*g*m*np.cos(x_bar)/np.sqrt((a*np.cos(x_bar))**2 + 1)
        
        # next, we assemble the QP
        # P_q:
        P_q = np.zeros((2*(N+1)+N,2*(N+1)+N))
        for k in range(0,N):
            P_q[2*k:2*k+2, 2*k:2*k+2] = 2.0*np.array([[self.Q_1, 0],[0, self.Q_2]])
        P_q[2*N:2*N+2, 2*N:2*N+2] = 2.0*np.array([[self.P_1, 0],[0, self.P_2]])
        for k in range(0,N):
            P_q[2*(N+1)+k,2*(N+1)+k] = 2.0*self.R
        # q_q:
        q_q = np.zeros((2*(N+1)+N))
        for k in range(0,N):
            q_q[2*k:2*k+2] = 2.0*np.matmul(np.array([[self.Q_1, 0],[0, self.Q_2]]),np.array([[x_bar-r_x],[v_bar-r_v]])).reshape((2))
        q_q[2*N:2*N+2] = 2.0*np.matmul(np.array([[self.P_1, 0],[0, self.P_2]]),np.array([[x_bar-r_x],[v_bar-r_v]])).reshape((2))
        for k in range(0,N):
            q_q[2*(N+1)+k] = 2.0*self.R*u_bar
        # G_q:
        G_q = np.zeros((2*N,2*(N+1)+N))
        for k in range(0,N):
            G_q[2*k,2*(N+1)+k] = 1.0
            G_q[2*k+1,2*(N+1)+k] = -1.0
        # h_q:
        h_q = np.zeros((2*N))
        for k in range(0,N):
            h_q[2*k] = u_max - u_bar
            h_q[2*k+1] = -u_min + u_bar
        # A_q:
        A_q = np.zeros((2+2*N,2*(N+1)+N))
        A_q[0:2,0:2] = np.eye(2)
        for k in range(0,N):
            A_q[2+2*k:2+2*k+2,2*k:2*k+2] = F
            A_q[2+2*k:2+2*k+2,2*k+2:2*k+4] = -np.eye(2)
            A_q[2+2*k:2+2*k+2,2*(N+1)+k:2*(N+1)+k+1] = G
        # b_q:
        b_q = np.zeros((2+2*N))
        b_q[0] = x-x_bar
        b_q[1] = v-v_bar
        x_q = solve_qp(P_q, q_q, G_q, h_q, A_q, b_q, solver='proxqp')
        #print("QP solution: x = {}".format(x_q))
        u = u_bar + x_q[2*(N+1)] # don't forget to apply delta to linearisation point...
        #print(u)
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
        self.controller = MpcController()
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
