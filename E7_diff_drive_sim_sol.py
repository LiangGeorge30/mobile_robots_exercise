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
CAR_WHEEL_RADIUS = 0.1    # car wheel radius
CAR_LENGTH = 1.4           # car length
NUM_LANDMARKS = 2         # number of landmarks to be used
CAM_F = 100                # camera focal length
CAM_W = 200                # image size (pixels)
CAM_C = 99.5               # image centre (pixels)
MATCH_OUTLIER_PROB = 0.05  # outlier probability of a measurement
ANGULAR_NOISE = 0.01       # noise on angle measurement (radians)
PIXEL_NOISE = 0.5          # noise on keypoint measurement (pixels)

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

class ParticleFilter:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.need_initialisation = True
        self.N_SAMPLES = 20
        self.particles = np.zeros((self.N_SAMPLES,3))
        self.weights = np.zeros((self.N_SAMPLES,1))
        return
    def reinitialise(self, x):
        self.particles[:,0] = np.random.normal(x[0], 0.1, self.N_SAMPLES)
        self.particles[:,1] = np.random.normal(x[1], 0.1, self.N_SAMPLES)
        self.particles[:,2] = np.random.normal(x[2], 0.02, self.N_SAMPLES)
        self.weights = np.ones((self.N_SAMPLES,1))*1.0/self.N_SAMPLES
        self.need_initialisation = False # since we are cheating a bit, this will always be successful...
        return
    def predict(self, u):
        dt = 1.0 / FPS
        v = (u[0]+u[1])*0.5*CAR_WHEEL_RADIUS
        omega = (u[1]-u[0])*CAR_WHEEL_RADIUS/CAR_WHEEL_BASE
        for p in range(0, self.N_SAMPLES): # this is not very Pythonic, feel free to improve...
            # predict mean:
            x_dot = np.cos(self.particles[p,2]) * v
            y_dot = np.sin(self.particles[p,2]) * v
            theta_dot = omega
            self.particles[p,:] += dt * np.array([x_dot, y_dot, theta_dot])
            # add noise:
            w_l = np.random.normal(0,ANGULAR_NOISE)
            w_r = np.random.normal(0,ANGULAR_NOISE)
            self.particles[p,0] += (w_l+w_r)*0.5*CAR_WHEEL_RADIUS
            self.particles[p,1] += (w_l+w_r)*0.5*CAR_WHEEL_RADIUS
            self.particles[p,2] += (w_r-w_l)*CAR_WHEEL_RADIUS/CAR_WHEEL_BASE
            # wraparound:
            self.particles[p,2] = self.particles[p,2] % (2 * np.pi)  # theta -> [0, 2pi) 
        return
    def update(self, z):
        # Bayesian update:
        for p in range(0, self.N_SAMPLES): # this is not very Pythonic, feel free to improve...
            x = self.particles[p]
            R_WB = np.array([[np.cos(x[2]), -np.sin(x[2])],
                             [np.sin(x[2]), np.cos(x[2])]])
            for (i, zi_tilde) in z:
                lm = self.landmarks[i,:]
                lm_B = np.matmul(np.transpose(R_WB), (lm-x[0:2]).reshape(2,1))
                zi = -CAM_F*lm_B[1]/lm_B[0]+CAM_C
                #print([zi_tilde, zi, normpdf(zi_tilde, zi, 0.5)])
                P = (1.0-MATCH_OUTLIER_PROB)*normpdf(zi_tilde, zi, PIXEL_NOISE) + MATCH_OUTLIER_PROB/CAM_W
                self.weights[p] *= P
        # re-normalise
        W = np.sum(self.weights)
        self.weights *= 1.0/W
        # re-sample:
        newparticles = self.particles
        for p in range(0, self.N_SAMPLES): # this is not very Pythonic, feel free to improve...
            s = np.random.uniform(p/self.N_SAMPLES,(p+1)/self.N_SAMPLES)
            cumsum = 0.0
            for p2 in range(0, self.N_SAMPLES): # this is not very Pythonic, feel free to improve...
                if s>cumsum and s<=cumsum+self.weights[p2]:
                    newparticles[p,:] = self.particles[p2,:] 
                    break
                cumsum += self.weights[p2]
        self.particles = newparticles # copy over
        self.weights = np.ones((self.N_SAMPLES,1))*1.0/self.N_SAMPLES # reset
        return
        
    def plotState(self):
        ax = plt.gca()
        for particle in self.particles:
            [x, y, theta] = particle
            R = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta), np.cos(theta)]])
            dxdy = np.matmul(R, np.array([-CAR_LENGTH / 2, - CAR_WIDTH / 2]))
            dx = np.matmul(R, np.array([[CAR_LENGTH], [0]]))
            car_rect = Rectangle((x + dxdy[0], y + dxdy[1]), angle=theta / np.pi * 180, width=CAR_LENGTH, height=CAR_WIDTH, color='gray')
            car_x = plt.plot([x,x+dx[0,0]],[y,y+dx[1,0]], color='gray')
            ax.add_patch(car_rect)
        return
        
class Ekf:
    def __init__(self, landmarks):
        self.need_initialisation = True
        self.landmarks = landmarks
        return
    def reinitialise(self, x):
        self.x = x
        self.P = np.array([[0.01, 0.0, 0.0],[0.0, 0.01, 0.0],[0.0, 0.0, 0.0004]])
        self.need_initialisation = False # since we are cheating a bit, this will always be successful...
        return
    def predict(self, u):
        dt = 1.0 / FPS
        v = (u[0]+u[1])*0.5*CAR_WHEEL_RADIUS
        omega = (u[1]-u[0])*CAR_WHEEL_RADIUS/CAR_WHEEL_BASE
        # predict mean:
        x_dot = np.cos(self.x[2]) * v
        y_dot = np.sin(self.x[2]) * v
        theta_dot = omega
        self.x += dt * np.array([x_dot, y_dot, theta_dot])
        self.x[2] = self.x[2] % (2 * np.pi)  # theta -> [0, 2pi) 
        # predict covariance:
        F = np.array([[1.0, 0.0, dt*v*np.sin(self.x[2])],
                      [0.0, 1.0, -dt*v*np.cos(self.x[2])],
                      [0.0, 0.0, 1.0]])
        L = np.array([[0.5*CAR_WHEEL_RADIUS, 0.5*CAR_WHEEL_RADIUS],
                      [0.5*CAR_WHEEL_RADIUS, 0.5*CAR_WHEEL_RADIUS],
                      [-CAR_WHEEL_RADIUS/CAR_WHEEL_BASE, CAR_WHEEL_RADIUS/CAR_WHEEL_BASE]])
        Q = np.array([[ANGULAR_NOISE**2, 0.0],[0.0, ANGULAR_NOISE**2]])
        self.P = np.matmul(np.matmul(F,self.P),np.transpose(F)) + np.matmul(np.matmul(L,Q),np.transpose(L))
        return
    def update(self, z):
        x = self.x
        R_WB = np.array([[np.cos(x[2]), -np.sin(x[2])],
                         [np.sin(x[2]), np.cos(x[2])]])
        for (i, zi_tilde) in z:
            lm = self.landmarks[i,:]
            lm_B = np.matmul(np.transpose(R_WB), (lm-x[0:2]).reshape(2,1))
            dx = lm[0]-x[0]
            dy = lm[1]-x[1]
            U = np.array([[CAM_F*lm_B[1,0]/lm_B[0,0]**2, -CAM_F/lm_B[0,0]]])
            Hi = np.zeros((1,3))
            Hi[0,0:2] = np.matmul(U,-np.transpose(R_WB))
            Hi[0,2] = np.matmul(U,np.array([[-np.sin(x[2])*dx+np.cos(x[2])*dy],[-np.cos(x[2])*dx-np.sin(x[2])*dy]]))
            zi = -CAM_F*lm_B[1,0]/lm_B[0,0]+CAM_C
            y = zi_tilde - zi # note: this is a scalar here
            S = np.matmul(np.matmul(Hi,self.P),np.transpose(Hi)) + PIXEL_NOISE*PIXEL_NOISE # this is therefore also a scalar
            K = np.matmul(self.P,np.transpose(Hi))/S
            if y**2/S > 100:
                continue # chi square test
            self.x += np.reshape(K*y,3)
            self.P -= np.matmul(K*Hi,self.P)
        return
        
    def plotState(self):
        ax = plt.gca()
        [x, y, theta] = self.x
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
        dxdy = np.matmul(R, np.array([-CAR_LENGTH / 2, - CAR_WIDTH / 2]))
        dx = np.matmul(R, np.array([[CAR_LENGTH], [0]]))
        car_rect = Rectangle((x + dxdy[0], y + dxdy[1]), angle=theta / np.pi * 180, width=CAR_LENGTH, height=CAR_WIDTH, color='gray')
        car_x = plt.plot([x,x+dx[0,0]],[y,y+dx[1,0]], color='gray')
        ax.add_patch(car_rect)
        ell = nSigmaEllipse(self.x[0:2],self.P[0:2,0:2], color='gray', n=5)
        ax.add_patch(ell)
        
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
        #self.estimator = ParticleFilter(self.landmarks) # select/initialise
        self.estimator = Ekf(self.landmarks) # select/initialise

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

        # plot estimate: either particle distribution or Gaussian...
        self.diffDriveSimulator.estimator.plotState()

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
