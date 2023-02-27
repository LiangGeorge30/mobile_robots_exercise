import dataclasses
import sys

from typing import List
from PyQt5 import QtWidgets, QtCore, QtGui

import numpy as np

X_SIZE = 600
Y_SIZE = 600

class Vertex:
    def __init__(self, x=np.array([0,0]), parent=-1):
        self.x = x
        self.parent = parent

class RRT:
    # TODO: implement RRT
    def __init__(self, x_min, x_max, y_min, y_max, in_collision):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.vertices = []
        self.startIdx = -1
        self.goalIdx = -1
        self.in_collision = in_collision
    def plan(self, start, goal, num_iter):
        self.vertices = []
        self.vertices.append(Vertex(start))
        self.startIdx = 0
        self.vertices.append(Vertex(goal, 0))
        self.goalIdx = 1

class RRTstar:
    # TODO: implement RRT*
    def __init__(self, x_min, x_max, y_min, y_max, in_collision):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.vertices = []
        self.startIdx = -1
        self.goalIdx = -1
        self.in_collision = in_collision
    def plan(self, start, goal, num_iter):
        self.vertices = []
        self.vertices.append(Vertex(start))
        self.startIdx = 0
        self.vertices.append(Vertex(goal, 0))
        self.goalIdx = 1

class InformedRRTstar:
    # TODO: implement informed RRT*
    def __init__(self, x_min, x_max, y_min, y_max, in_collision):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.vertices = []
        self.startIdx = -1
        self.goalIdx = -1
        self.in_collision = in_collision
    def plan(self, start, goal, num_iter):
        self.vertices = []
        self.vertices.append(Vertex(start))
        self.startIdx = 0
        self.vertices.append(Vertex(goal, 0))
        self.goalIdx = 1

@dataclasses.dataclass(frozen=True)
class Rect2D:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Rectangle either has no area or wrong assignment")

    def is_in(self, x: int, y: int) -> bool:
        return self.x_min < x < self.x_max and self.y_min < y < self.y_max

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

class Environment2D:

    def __init__(self) -> None:
        self.obstacles: List[Rect2D] = []
        self.borders = Rect2D(0, 0, X_SIZE, Y_SIZE)
        self.start = (10, 10)
        self.goal = (20, 20)
        self.planner = RRT(0, X_SIZE, 0, Y_SIZE, self.in_collision)

    def plan(self, num_iter):
        self.planner.plan(np.array([self.start[0], self.start[1]]), 
                          np.array([self.goal[0], self.goal[1]]), num_iter)

    def in_collision(self, x: int, y: int) -> bool:
        if not self.borders.is_in(x, y):
            return True
        return any(obstacle.is_in(x, y) for obstacle in self.obstacles)

    def add_obstacle(self, x_min: int, y_min: int, x_max: int, y_max: int):
        box2d = Rect2D(x_min, y_min, x_max, y_max)
        self.obstacles.append(box2d)

    def clear(self):
        self.obstacles = []
        self.planner.vertices = []
        self.planner.goalIdx = -1
        self.planner.startId = -1


class PlanningVisualizationWidget(QtWidgets.QWidget):
    
    class DRAWING_STATE:
        DRAW_START = 0
        DRAW_GOAL = 1
        DRAW_BOX = 2

    def __init__(self, environment: Environment2D):
        super(PlanningVisualizationWidget, self).__init__()

        self.env = environment
        self.cache_box_start = None
        self.cache_box_end = None
        self.draw_status = self.DRAWING_STATE.DRAW_BOX

        width = self.env.borders.width
        height = self.env.borders.height
        self.setGeometry(30, 30, width, height)
        self.setFixedSize(width, height)
        self.setWindowTitle("Planning Visualization Tool")

        self.area = QtWidgets.QWidget(self)
        self.draw_box = QtWidgets.QPushButton("draw")
        self.draw_box.clicked.connect(self.set_draw_box)
        self.start_button = QtWidgets.QPushButton("start ❌")
        self.start_button.clicked.connect(self.set_draw_start)
        self.goal_button = QtWidgets.QPushButton("goal 🏁")
        self.goal_button.clicked.connect(self.set_draw_goal)
        
        # Create textbox
        self.label = QtWidgets.QLabel("No. Iter:")
        #self.label.setAlignment(QtCore.Qt.AlignCenter)
        #self.label.move(20, 40)
        self.label.resize(50,20)
        self.textbox = QtWidgets.QLineEdit(self)
        self.textbox.setText("100")
        self.iter_validator = QtGui.QIntValidator(1,1000000000)
        self.textbox.setValidator(self.iter_validator)
        #self.textbox.move(20, 100)
        self.textbox.resize(80,20)

        self.plan_button = QtWidgets.QPushButton("plan")
        #self.plan_button.setStyleSheet("background-color: red")
        self.plan_button.clicked.connect(self.planning_callback)
        self.clear_button = QtWidgets.QPushButton("clear")
        #self.clear_button.setStyleSheet("background-color: yellow")
        self.clear_button.clicked.connect(self.clear_environment)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.area, 0, 0)
        layout.addWidget(self.start_button, 1, 0)
        layout.addWidget(self.goal_button, 1, 1)
        layout.addWidget(self.draw_box, 1, 2)
        layout.addWidget(self.plan_button, 1, 3)
        layout.addWidget(self.clear_button, 1, 4)
        layout.addWidget(self.label, 1, 5)
        layout.addWidget(self.textbox, 1, 6)

        self.setLayout(layout)
        self.show()

    ###############################################################################################
    # Button and Mouse Callbacks ##################################################################
    ###############################################################################################
    def set_draw_start(self):
        self.draw_status = self.DRAWING_STATE.DRAW_START
        
    def set_draw_goal(self):
        self.draw_status = self.DRAWING_STATE.DRAW_GOAL
        
    def set_draw_box(self):
        self.draw_status = self.DRAWING_STATE.DRAW_BOX
    
    def mousePressEvent(self, event):
        # All pressable points are in the borders in the environment by construction.
        if self.draw_status == self.DRAWING_STATE.DRAW_BOX:
            self.cache_box_start = event.pos()
            self.cache_box_end = event.pos()

        elif self.draw_status == self.DRAWING_STATE.DRAW_START:
            self.env.start = (event.pos().x(), event.pos().y())

        elif self.draw_status == self.DRAWING_STATE.DRAW_GOAL:
            self.env.goal = (event.pos().x(), event.pos().y())

        self.update()

    def mouseMoveEvent(self, event):
        # All pressable points are in the borders in the environment by construction.
        if self.draw_status == self.DRAWING_STATE.DRAW_BOX:
            self.cache_box_end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        # All pressable points are in the borders in the environment by construction.
        if self.draw_status == self.DRAWING_STATE.DRAW_BOX:
            if self.cache_box_start != self.cache_box_end:
                x_min = min(self.cache_box_start.x(), self.cache_box_end.x())
                y_min = min(self.cache_box_start.y(), self.cache_box_end.y())
                x_max = max(self.cache_box_start.x(), self.cache_box_end.x())
                y_max = max(self.cache_box_start.y(), self.cache_box_end.y())
                if x_min == x_max:
                    x_max += 1
                if y_min == y_max:
                    y_max += 1
                self.env.add_obstacle(x_min, y_min, x_max, y_max)
            self.cache_box_start = None
            self.cache_box_end = None
        self.update()

    def clear_environment(self):
        self.env.clear()
        self.update()

    def planning_callback(self):
        num_iter = int(round(float(self.textbox.text())))
        self.textbox.setText(str(num_iter))
        self.env.plan(num_iter)
        self.update()

    ###############################################################################################
    # Painting ####################################################################################
    ###############################################################################################
    def paintEvent(self, event):
        qp = QtGui.QPainter(self)

        # If currently drawing a box, then draw it in slightly opaque red color.
        br = QtGui.QBrush(QtGui.QColor(100, 100, 10, 40))
        qp.setBrush(br)
        if self.draw_status == self.DRAWING_STATE.DRAW_BOX:
            if self.cache_box_start is not None and self.cache_box_end is not None:
                rect_current = QtCore.QRect(self.cache_box_start, self.cache_box_end)
                qp.drawRect(rect_current)

        # Draw boxes in environment. 
        br = QtGui.QBrush(QtGui.QColor(10, 10, 10, 255))
        qp.setBrush(br)
        for rect in self.env.obstacles:
            box_lower_left = QtCore.QPoint(rect.x_min, rect.y_min)
            box_upper_right = QtCore.QPoint(rect.x_max, rect.y_max)
            rect = QtCore.QRect(box_lower_left, box_upper_right)
            qp.drawRect(rect)

        # Draw planned trajectory.
        pn = QtGui.QPen(QtGui.QColor(127, 127, 127))
        qp.setPen(pn)
        path_lines = []
        # first full tree
        for k in range(0, len(self.env.planner.vertices)):
            if self.env.planner.vertices[k].parent >=0:
                point_km1 = QtCore.QPoint(*np.round(self.env.planner.vertices[self.env.planner.vertices[k].parent].x).astype(int))
                point_k = QtCore.QPoint(*np.round(self.env.planner.vertices[k].x).astype(int))
                path_segment = QtCore.QLineF(point_km1, point_k)
                path_lines.append(path_segment)
        qp.drawLines(path_lines)
        # now also trace the final path
        pn = QtGui.QPen(QtGui.QColor(0, 31, 255))
        qp.setPen(pn)
        path_lines = []
        k = self.env.planner.goalIdx
        if k >= 0:
            parent = self.env.planner.vertices[k].parent
            while parent >= 0:
                point_km1 = QtCore.QPoint(*np.round(self.env.planner.vertices[parent].x).astype(int))
                point_k = QtCore.QPoint(*np.round(self.env.planner.vertices[k].x).astype(int))
                path_segment = QtCore.QLineF(point_km1, point_k)
                path_lines.append(path_segment)
                k = parent
                parent = self.env.planner.vertices[parent].parent
            qp.drawLines(path_lines)

        # Draw start and goal point in envioronment.
        goal_rect = QtCore.QRect(self.env.goal[0] - 10, self.env.goal[1] - 10, 20, 20)
        qp.drawText(goal_rect, QtCore.Qt.AlignCenter, "🏁")
        start_rect = QtCore.QRect(self.env.start[0] - 10, self.env.start[1] - 10, 20, 20)
        qp.drawText(start_rect, QtCore.Qt.AlignCenter, "❌ ")

        qp.end()


if __name__ == '__main__':
    env = Environment2D()
    app = QtWidgets.QApplication(sys.argv)
    window = PlanningVisualizationWidget(env)
    window.show()
    sys.exit(app.exec_())
