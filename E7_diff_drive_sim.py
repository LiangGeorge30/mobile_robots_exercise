import matplotlib.pyplot as plt
import numpy as np
import PyQt5.Qt as Qt
import PyQt5.QtCore as QtCore
import sys

from collections import namedtuple
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse, Rectangle
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout


FPS = 20                   # simulation frames per second
X_MIN, X_MAX = -10, 10     # simulation environment size along x
Y_MIN, Y_MAX = -10, 10     # simulation environment size along y
CAR_WIDTH = 1.1            # car width
CAR_WHEEL_BASE = 1.2       # car wheel base
CAR_WHEEL_RADIUS = 0.1     # car wheel radius
CAR_LENGTH = 1.4           # car length
NUM_LANDMARKS = 10         # number of landmarks to be used
CAM_F = 100                # camera focal length
CAM_W = 200                # image size (pixels)
CAM_C = 99.5               # image centre (pixels)
MATCH_OUTLIER_PROB = 0.05  # outlier probability of a measurement
ANGULAR_NOISE = 0.01       # noise on angle measurement (radians)
PIXEL_NOISE = 0.5          # noise on keypoint measurement (pixels)


class ParticleFilter:
    def __init__(self):
        self.need_initialisation = True
        # TODO: implement me...
        return
    def reinitialise(self, x):
        # TODO: implement me...
        # if successful: need_initialisation = False
        return
    def predict(self, u):
        # TODO: implement me...
        return
    def update(self, z):
        # TODO: implement me...
        return
        
        
class Ekf:
    def __init__(self):
        self.need_initialisation = True
        # TODO: implement me...
        return
    def reinitialise(self, x):
        # TODO: implement me...
        # if successful: need_initialisation = False
        return
    def predict(self, u):
        # TODO: implement me...
        return
    def update(self, z):
        # TODO: implement me...
        return
        
# helper function to plot a 2D uncertainty ellipse (n-sigma)
def nSigmaEllipse(mean, cov, color, n=3):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(mean[0], mean[1]),
                  width=lambda_[0]*n*2, height=lambda_[1]*n*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  color=color, fill=False)
    return ell

class DiffDriveSimulator:

    def __init__(self, x: float, y: float, theta: float):
        self.w = CAR_WHEEL_BASE # wheel base
        self.r = CAR_WHEEL_RADIUS # wheel radius
        self.num_landmarks = NUM_LANDMARKS # number of landmarks to generate
        self.pose = np.array([x, y, theta])
        self.controls = np.zeros(2)  # speed, rotation rate
        self.previous_controls = np.zeros(2)  # speed, rotation rate
        self.key_state = np.zeros(2) # fwd/bwd, rot-left/rot-right
        self.landmarks = np.random.uniform(low=[X_MIN, Y_MIN], high=[X_MAX, Y_MAX], size=(self.num_landmarks,2))
        self.estimator = ParticleFilter() # TODO: select/initialise

    def update(self, dt: float):
        
        # some higher-order dynamics from the key state
        self.controls = self.previous_controls* 0.8 + 0.2*self.key_state*np.array([2.0, np.pi / 4.0])
        self.previous_controls = self.controls
        
        v, omega = self.controls
        _, _, theta = self.pose
        
        # kinematics using Euler-forward discretisation
        x_dot = np.cos(theta) * v
        y_dot = np.sin(theta) * v
        theta_dot = omega
        self.pose = self.pose + dt * np.array([x_dot, y_dot, theta_dot])
        self.pose[2] = self.pose[2] % (2 * np.pi)  # theta -> [0, 2pi)
        
        # simulate the wheel speed measurements (with some noise, if turning)
        wheel_rotvel_diff = omega*self.w/self.r
        wheel_rotvel_mean = v/self.r
        u = np.zeros(2)
        u[0] = wheel_rotvel_mean - 0.5*wheel_rotvel_diff
        u[1] = wheel_rotvel_mean + 0.5*wheel_rotvel_diff
        if abs(u[0]) > 0.0001:
            u[0] += np.random.normal(0,ANGULAR_NOISE/dt)
        if abs(u[1]) > 0.0001:
            u[1] += np.random.normal(0,ANGULAR_NOISE/dt)
        
        # generate measurements
        z = []
        for i in range(0, len(self.landmarks)):
            lm = self.landmarks[i,:]
            R_WB = np.array([[np.cos(self.pose[2]), -np.sin(self.pose[2])],
                             [np.sin(self.pose[2]), np.cos(self.pose[2])]])
            lm_B = np.matmul(np.transpose(R_WB), (lm-self.pose[0:2]).reshape(2,1))
            if lm_B[0]>1.0e-10 :
                cam_u = -CAM_F*lm_B[1]/lm_B[0]+CAM_C
                if cam_u > 0.5 and cam_u < CAM_W-0.5:
                    zi = cam_u + np.random.normal(0,PIXEL_NOISE)
                    o = np.random.uniform(0,1)
                    if o<MATCH_OUTLIER_PROB:
                        zi = np.random.uniform(-0.5,CAM_W-0.5)
                    z.append((i, zi))
        
        # call prediction and update functions on your estimator
        if self.estimator.need_initialisation:
            self.estimator.reinitialise(self.pose)
        else:
            self.estimator.predict(u)
            self.estimator.update(z)
                
    @property
    def speed(self) -> float:
        return self.controls[0]

    @property
    def rotation_rate(self) -> float:
        return self.controls[1]

    def __str__(self) -> str:
        return f"DiffDriveSimulator(pose={self.pose})"


class Window(QWidget):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.diffDriveSimulator = DiffDriveSimulator(x=0, y=0, theta=0)
        self.k = 0

        # set the layout
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # screen_size = self.screen().geometry().size()
        # min_size = min(screen_size.width(), screen_size.height())
        # self.resize(QtCore.QSize(int(min_size / 2), int(min_size / 2)))

        # Timer for updating the view, with a delta t of 1s / fps between frames.
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000.0 / FPS))  # in milliseconds

    def update(self):
        self.diffDriveSimulator.update(dt=1 / FPS)
        self.plot()
        self.k += 1

    ####################################################################################################################
    # Interactions #####################################################################################################
    ####################################################################################################################
    def keyPressEvent(self, event):
        key_pressed = event.key()
        
        if key_pressed == QtCore.Qt.Key.Key_Right:
            self.diffDriveSimulator.key_state[1] = -1
        if key_pressed == QtCore.Qt.Key.Key_Left:
            self.diffDriveSimulator.key_state[1] = 1

        if key_pressed == QtCore.Qt.Key.Key_Up:
            self.diffDriveSimulator.key_state[0] = 1
        if key_pressed == QtCore.Qt.Key.Key_Down:
            self.diffDriveSimulator.key_state[0] = -1

        if key_pressed == QtCore.Qt.Key.Key_Escape:
            self.close()
            
    def keyReleaseEvent(self, event):
        key_pressed = event.key()

        if key_pressed == QtCore.Qt.Key.Key_Right:
            self.diffDriveSimulator.key_state[1] = 0
        if key_pressed == QtCore.Qt.Key.Key_Left:
            self.diffDriveSimulator.key_state[1] = 0

        if key_pressed == QtCore.Qt.Key.Key_Up:
            self.diffDriveSimulator.key_state[0] = 0
        if key_pressed == QtCore.Qt.Key.Key_Down:
            self.diffDriveSimulator.key_state[0] = 0

    ####################################################################################################################
    # Rendering ########################################################################################################
    ####################################################################################################################
    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)

        # TODO: plot estimate: either particle distribution or Gaussian...

        x, y, theta = self.diffDriveSimulator.pose
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
        dxdy = np.matmul(R, np.array([-CAR_LENGTH / 2, - CAR_WIDTH / 2]))
        dx = np.matmul(R, np.array([[CAR_LENGTH], [0]]))

        car_rect = Rectangle((x + dxdy[0], y + dxdy[1]), angle=theta / np.pi * 180, width=CAR_LENGTH, height=CAR_WIDTH)
        car_x = plt.plot([x,x+dx[0,0]],[y,y+dx[1,0]], color='red')
        ax.add_patch(car_rect)
        
        plt.scatter(self.diffDriveSimulator.landmarks[:,0], self.diffDriveSimulator.landmarks[:,1])

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
