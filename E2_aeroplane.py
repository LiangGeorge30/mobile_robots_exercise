#written by Stefan Leutenegger, TU Munich, 27.10.2021
import vedo
import numpy as np

#vedo.embedWindow('ipyvtk')

class Viewer:
    
    def __init__(self):
        self.plot = vedo.Plotter(axes=3) #with frame E axis
        self.plot.camera.SetPosition([-2, 10, -5])
        self.plot.camera.SetViewUp([0.2, -1.0, 0.0])
        self.plot.camera.SetFocalPoint([0, 0, 0])
        
        #load a nice aeroplane
        self.aeroplane = vedo.load(vedo.dataurl+"cessna.vtk")
        
        #put a sensible bounding box aorund it
        self.box = vedo.Box([0,0,0], 6, 6, 6).wireframe()
        
        #since the object was defined with "robotics" standard z-up, let's remember to fix this later
        #through a separate "Object" frame.
        self.T_BO = np.array([[1.0,0.0,0.0,0.0],[0.0,-1.0,0.0,0.0],[0.0,0.0,-1.0,0.0],[0.0,0.0,0.0,1.0]])
      
        #sliders
        self.R = 0.0
        self.P = 0.0
        self.Y = 0.0
        self.plot.addSlider2D(self.sliderR, -180, 180, value=0,
               pos=1, title="roll deg")
        self.plot.addSlider2D(self.sliderP, -90, 90, value=0,
               pos=2, title="pitch deg")
        self.plot.addSlider2D(self.sliderY, -180, 180, value=0,
               pos=3, title="yaw deg")
    
    def render(self, r, p, y):
        #TODO: compute transform and apply before plotting...!
        T_EB = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
        self.aeroplane.applyTransform(np.matmul(T_EB,self.T_BO))
        
        #TODO: construct coordinate axes of the plane-body-frame B
        #something like this: 
        #Btx = vedo.Cylinder(pos=((0, 0, 0),(3, 0, 0)), r=0.01, c='red')
        
        #now let's plot everything
        plt = self.plot.show(self.aeroplane, self.box, resetcam=False, interactive=1) #TODO: add the different axes to the plot, too!
        return plt
               
    def sliderR(self, widget, event):
        value = widget.GetRepresentation().GetValue()
        self.R = value
        self.render(self.R, self.P, self.Y)
    def sliderP(self, widget, event):
        value = widget.GetRepresentation().GetValue()
        self.P = value
        self.render(self.R, self.P, self.Y)
    def sliderY(self, widget, event):
        value = widget.GetRepresentation().GetValue()
        self.Y = value
        self.render(self.R, self.P, self.Y)
        
#start the interactive viewer
viewer = Viewer()
viewer.render(0, 0, 0)