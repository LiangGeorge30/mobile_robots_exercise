# written by Stefan Leutenegger, TU Munich, November 2021

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# helper class that just does the radial-tangentialDistortion
class RadialTangentialDistortion:
    def __init__(self, k1, k2, p1, p2):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        
    def distort(self, uUndistorted):
        uDistorted = uUndistorted # TODO: implement!
        return uDistorted
    
    def undistort(self, uDistorted):
        num_points = uDistorted.shape[0]
        uUnDistorted = np.zeros_like(uDistorted)
        # make it work for many points:
        for k in range(num_points):
            uDistortedk = uDistorted[k, :]
            diff = lambda uUndistortedk, uDistorted_k : np.linalg.norm(self.distort(uUndistortedk)-uDistortedk)
            res = minimize(diff, uDistortedk, args=(uDistortedk), method='Nelder-Mead', tol=1e-6)
            uUnDistorted[k, :] = res.x
        return uUnDistorted
        
# now the pinhole camera class
class PinholeCamera:
    def __init__(self, width, height, f1, f2, c1, c2, distortion):
        self.width = width
        self.height = height
        self.f1 = f1
        self.f2 = f2
        self.c1 = c1
        self.c2 = c2
        self.distortion = distortion
    def project(self, x):
        # TODO
        return []
        
    def p(self, x):
        # TODO
        return []
        
    def d(self, x_dash):
        return self.distortion.distort(x_dash)
    
    def d_inverse(self, x_dash):
        return self.distortion.undistort(x_ddash)
    
    def k(self, x_ddash):
        # TODO
        return []

# now test with a projected cube
b = 1.0 # sidelength
z_distance = 3.0 # distance from the camera along the z-axis
spacing = np.array([np.linspace(-b/2.0, b/2.0, 50)])
eb0 = np.array([[0],[-b/2.0],[-b/2.0 + z_distance]]) + np.array([[1.0],[0.0],[0.0]]).dot(spacing)
edges = [eb0]
# TODO: generate the other 11 edges equivalently
#edges = [eb0, eb1, eb2, eb3, es1, es2, es3, es4, et0, et1, et2, et3]

# create a plausible pinhole camera model, VGA resolution
pinholeCamera = PinholeCamera(640, 480, 450, 450, 319.5, 239.5,
                              RadialTangentialDistortion(-0.3, 0.1, -0.0001, -0.00005))

for edge in edges:
    for column in edge.T:
        u = pinholeCamera.project(column.T)
    # TODO: plot the whole edge

# here is a unit test
pinholeCamera = PinholeCamera(640, 480, 450, 450, 319.5, 239.5,
                              RadialTangentialDistortion(-0.3, 0.1, -0.0001, -0.00005))
for i in range(0,1000):
    print(i)
    # TODO: generate random visible point in image
    # TODO: back-project and assign random distance
    # TODO: project again
    # TODO: check the projection is the same as the generated initial image point
