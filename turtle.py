import numpy as np
import matplotlib
import matplotlib.pyplot as mpl
import scipy.optimize

"""
perimeter = [(heading in degrees, distance), ...]
x = easting
y = northing
"""
def polar2xy(perimeter):
    xy = []
    for p in perimeter:
        x = p[1] * np.sin(p[0]*np.pi/180)
        y = p[1] * np.cos(p[0]*np.pi/180)
        xy.append((x,y))
    xy = np.array(xy)
    x = np.cumsum(xy[:,0])
    y = np.cumsum(xy[:,1])
    return x, y

"""
Strings are of the form:
"N 77:21:00 W"
"""
def string_to_angle(s):
    tokens = s.split()
    assert(tokens[0] in ("N", "S"))
    assert(tokens[2] in ("E", "W"))
    if tokens[0].upper() == "N":
        origin = 0
        if tokens[2] == "E":
            sign = 1
        else:
            sign = -1
    else:
        origin = 180
        if tokens[2] == "E":
            sign = -1
        else:
            sign = 1

    degrees, minutes, seconds = [float(t) for t in tokens[1].split(':')]
    angle = origin + sign * (degrees + minutes / 60.0 + seconds / 3600.0)
    angle = angle % 360
    return angle


def test1():
    print(string_to_angle("N 00:00:00 E"))
    print(string_to_angle("N 45:00:00 E"))
    print(string_to_angle("N 90:00:00 E"))
    print(string_to_angle("S 45:00:00 E"))
    print(string_to_angle("S 00:00:00 E"))
    print(string_to_angle("S 45:00:00 W"))
    print(string_to_angle("S 90:00:00 W"))
    print(string_to_angle("N 45:00:00 W"))
    print()
    print(string_to_angle("N 00:01:00 E") * 60)
    print(string_to_angle("N 00:00:01 E") * 3600)


def law_of_cosines(a, b, c):
        # Returns the angle (in degrees) opposite side a.
        return np.arccos((-a**2 + b**2 + c**2) / (2*b*c)) * 180 / np.pi

def ang_dist(p1, p2, p3):
        """
        p1 is backsite
        p2 is the point you're on
        p3 is the point you're going to
        """
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p2[0] - p3[0]
        dy2 = p2[1] - p3[1]
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        angle = angle2 - angle1
        angle = angle * 180 / np.pi
        distance = np.sqrt(dx2**2 + dy2**2)
        return angle, distance

def distance(p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

def extend(p1, p2, d23, angle=0, ccw=[]):
        """
        p1 ----- p2 ------------ p3
                                        d23

        Find point p3 such that it is distance d23 beyond p2
        If ccw is a number, it is the angle in degrees at p2, measured
        counter-clockwise (as seen from above, looking downward) from p1 to p3.
        """

        direction = p2 - p1
        direction = direction / np.sqrt(direction[0]**2 + direction[1]**2)

        if ccw != []:
                a1 = (180.0 + ccw) * np.pi / 180
        else:
                a1 = angle * np.pi / 180

        rotate = np.array([[np.cos(a1), -np.sin(a1)],
                                             [np.sin(a1), np.cos(a1)]])
        direction = np.dot(rotate, direction)

        p3 = direction * d23 + p2

        return p3


class Turtle:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.heading = 0
        self.points = {'origin': (0,0,0)}
        self.rotation = 0

    def set_rotation(self, angle):
        self.rotation = string_to_angle(angle)

    def line(self, angle, distance, color='k', name=None, **kwargs):
        heading = string_to_angle(angle) + self.rotation
        x = self.x
        y = self.y
        dx = distance * np.sin(heading*np.pi/180)
        dy = distance * np.cos(heading*np.pi/180)
        out = mpl.plot((x, x+dx), (y, y+dy), color=color, **kwargs)
        self.x += dx
        self.y += dy
        self.heading = heading
        if name is not None:
            self.points[name] = (self.x, self.y, self.heading)
        return out

    def arc(self, r, length, left=True, ax=None, name=None, **kwargs):
        hx = np.sin(self.heading * np.pi / 180)
        hy = np.cos(self.heading * np.pi / 180)

        if left:
            center_dx = -r * hy
            center_dy = r * hx
        else:
            center_dx = r * hy
            center_dy = -r * hx

        center_x = self.x + center_dx
        center_y = self.y + center_dy

        delta_degrees = (180 / np.pi) * length / r
        print(delta_degrees)
        if left:
            degrees1 = self.heading - 0.5 * delta_degrees
            degrees2 = self.heading - delta_degrees
        else:
            degrees1 = self.heading + 0.5 * delta_degrees
            degrees2 = self.heading + delta_degrees

        crow_dist = 2 * r * np.sin(0.5 * delta_degrees * np.pi / 180)
        new_x = self.x + crow_dist * np.sin(degrees1 * np.pi / 180)
        new_y = self.y + crow_dist * np.cos(degrees1 * np.pi / 180)

        theta1 = (180 / np.pi) * np.arctan2(self.y - center_y, self.x - center_x)
        theta2 = (180 / np.pi) * np.arctan2(new_y - center_y, new_x - center_x)
        if left == False:
            temp = theta2
            theta2 = theta1
            theta1 = temp
        curve = matplotlib.patches.Arc((center_x, center_y), 2*r, 2*r, angle=0.0, theta1=theta1, theta2=theta2, **kwargs)
        if ax is None:
            ax = mpl.gca()
        ax.add_patch(curve)
        #mpl.plot((center_x, self.x, new_x), (center_y, self.y, new_y))
        self.x = new_x
        self.y = new_y
        self.heading = degrees2
        if name is not None:
            self.points[name] = (self.x, self.y, self.heading)
        return curve

    def dot(self, fmt='sk', ax=None, **kwargs):
        if ax is None:
            ax = mpl.gca()
        ax.plot((self.x,), (self.y,), fmt, **kwargs)

    def goto(self, name):
        assert(name in self.points)
        self.x = self.points[name][0]
        self.y = self.points[name][1]
        self.heading = self.points[name][2]

    def goto_xy(self, x, y, heading='N 00:00:00 E'):
        self.x = x
        self.y = y
        self.heading = string_to_angle(heading)

