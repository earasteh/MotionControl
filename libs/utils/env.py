"""
This file contains the class definition of the environment the vehicle is moving in (road, obstacles, etc.)
"""
import os
import numpy as np
from libs.utils.cubic_spline_interpolator import generate_cubic_spline
import pandas as pd
import matplotlib.pyplot as plt


class Path:
    def __init__(self, x, y, ds=0.05):
        self.ds = ds
        self.px, self.py, self.pyaw, _, self.s = generate_cubic_spline(x, y, self.ds)

    def create_fromcsv(self, dir):
        df = pd.read_csv(dir)
        x = df['X-axis'].values
        y = df['Y-axis'].values
        self.px, self.py, self.pyaw, _, self.s = generate_cubic_spline(x, y, self.ds)


class Env:
    def __init__(self):
        self.x_range = (0, 2000)
        self.y_range = (0, 2000)
        # self.road_boundries = self.road_boundries(path1, 1)
        # Locations of road boundries
        self.bound_xr = []
        self.bound_yr = []
        self.bound_xl = []
        self.bound_yl = []
        # The middle of the road for motion planning purposes
        self.xm = []
        self.ym = []
        # Initializing a path
        self.path = Path([0, 1], [0, 1], 0.05)
        # Obstacles
        self.obstacle_x = []
        self.obstacle_y = []
        self.obstacle_xy = []

    def road_bound_add(self, xlist, ylist, side):
        if side == 'right':
            self.bound_xr += xlist
            self.bound_yr += ylist
        elif side == 'left':
            self.bound_xl += xlist
            self.bound_yl += ylist
        else:
            raise Exception('Choose either right or left as the last argument')

    def road_middle(self):
        self.xm = (np.asarray(self.bound_xr) + np.asarray(self.bound_xl)) / 2
        self.ym = (np.asarray(self.bound_yr) + np.asarray(self.bound_yl)) / 2
        self.path = Path(self.xm, self.ym, 0.05)

    def obstacle_add(self, xlist, ylist):
        self.obstacle_x += xlist
        self.obstacle_y += ylist
        xy = []

        for i in range(len(xlist)):
            xy.append([xlist[i], ylist[i]])

        self.obstacle_xy = xy

    @staticmethod
    def strightline(x_start, x_end, y, ds):
        X = [x for x in np.arange(x_start, x_end, ds)]
        Y = [y] * len(X)
        return X, Y

    @staticmethod
    def circle(center, radius, ds, arc=None):
        """
        :param center: [xc, yc], coordinates of the center
        :param radius: radius
        :param arc: [0, 2*pi] gives a full-circle, [theta1, theta2] gives an arc with those specified initial and end angles
        :param ds:
        :return: X, Y which are the list of x, y locations
        """
        if arc is None:
            arc = [0, 2 * np.pi]

        theta1, theta2 = arc
        xc, yc = center
        X = [radius * np.cos(t) + xc for t in np.arange(theta1, theta2 + ds, ds)]
        Y = [radius * np.sin(t) + yc for t in np.arange(theta1, theta2 + ds, ds)]
        return X, Y

    @staticmethod
    def box(corner, width, length, ds):
        """
        Creates a box
        :param corner:
        :param width:
        :param length:
        :param ds:
        :return:
        """
        xc1, yc1 = corner
        xc2 = xc1 + length
        yc2 = yc1 + width

        X = []
        Y = []

        x1, y1 = Env.strightline(xc1, xc2, yc1, ds)
        X += x1
        Y += y1

        y2 = [y for y in np.arange(yc1, yc2, ds)]
        x2 = [xc2] * len(y2)
        X += x2
        Y += y2

        x3, y3 = Env.strightline(xc2, xc1, yc2, -ds)
        X += x3
        Y += y3

        y4 = [y for y in np.arange(yc1, yc2, ds)]
        x4 = [xc1] * len(y4)
        X += x4
        Y += y4

        return X, Y


# Construct the road using lines and circles
world = Env()

x1, y1 = Env.strightline(0, 100, 0, world.path.ds)
world.road_bound_add(x1, y1, 'right')
x2, y2 = Env.circle([100, 50], 50, world.path.ds, [-np.pi / 2, np.pi / 2])
world.road_bound_add(x2, y2, 'right')
x3, y3 = Env.strightline(100, 0, 100, -world.path.ds)
world.road_bound_add(x3, y3, 'right')
x4, y4 = Env.circle([0, 50], 50, world.path.ds, [np.pi / 2, 3 * np.pi / 2])
world.road_bound_add(x4, y4, 'right')

x1, y1 = Env.strightline(0, 100, 20, world.path.ds)
world.road_bound_add(x1, y1, 'left')
x2, y2 = Env.circle([100, 50], 30, world.path.ds, [-np.pi / 2, np.pi / 2])
world.road_bound_add(x2, y2, 'left')
x3, y3 = Env.strightline(100, 0, 80, -world.path.ds)
world.road_bound_add(x3, y3, 'left')
x4, y4 = Env.circle([0, 50], 30, world.path.ds, [np.pi / 2, 3 * np.pi / 2])
world.road_bound_add(x4, y4, 'left')
# find the middle of the road
world.road_middle()


straight = Env()
x1, y1 = Env.strightline(0, 400, 0, world.path.ds)
straight.road_bound_add(x1, y1, 'right')
x1, y1 = Env.strightline(0, 400, 20, world.path.ds)
straight.road_bound_add(x1, y1, 'left')
straight.road_middle()

def main():
    plt.figure()
    # plt.plot(path1.px, path1.py)
    plt.plot(world.bound_xr, world.bound_yr)
    plt.plot(world.bound_xl, world.bound_yl)
    plt.plot(world.xm, world.ym, linestyle='dashed', color='green')
    plt.fill(world.obstacle_x, world.obstacle_y, color='red')
    plt.axis('equal')
    plt.show()

    print(type(np.array(world.obstacle_xy)))


if __name__ == "__main__":
    main()
