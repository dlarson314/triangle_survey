import re
from collections import deque

import numpy as np
import matplotlib.pyplot as mpl
import scipy.optimize



def law_of_cosines(a, b, c): 
    # Returns the angle (in degrees) opposite side a.
    return np.arccos((-a**2 + b**2 + c**2) / float(2*b*c)) * 180 / np.pi


def solve_triangle(p1, p2, d23, d31):
    """ 
    p1 is point 1
    p2 is point 2
    d12 is the distance from point 1 to point 2
    d23 is the distance from point 2 to point 3
    d31 is the distance from point 3 to point 1

          p3
         /  \
    d31 /    \  d23
       /      \
      /        \
    p1 ------- p2
         d12

    return value is p3, the third point
    """

    d12 = np.sqrt(np.sum((p1 - p2)**2))

    # a1 is angle at point p1.
    a1 = law_of_cosines(d23, d12, d31) * np.pi/180

    rotate = np.array([[np.cos(a1), -np.sin(a1)],
                       [np.sin(a1), np.cos(a1)]])

    direction = np.dot(rotate, p2 - p1)
    direction = direction / np.sqrt(direction[0]**2 + direction[1]**2)
    p3 = direction * d31 + p1

    d12b = np.sqrt(np.sum((p1 - p2)**2))
    d23b = np.sqrt(np.sum((p2 - p3)**2))
    d31b = np.sqrt(np.sum((p3 - p1)**2))
    assert(abs(d12b - d12) < 1e-10)
    assert(abs(d23b - d23) < 1e-10)
    assert(abs(d31b - d31) < 1e-10)

    return p3


def add_triangle(points, triangle):
    triple = deque(triangle[0:3])
    sides = deque(triangle[3:6])

    if triple[2] in points:
        triple.rotate(1)
        sides.rotate(1)
    if triple[2] in points:
        triple.rotate(1)
        sides.rotate(1)

    if triple[2] in points:
        # If all three points are already solved, don't do anything else.
        return points

    print 'adding: ', triple, sides

    points[triple[2]] = solve_triangle(
        points[triple[0]],
        points[triple[1]],
        sides[1],
        sides[2])

    return points


def dist(a,b):
    return np.sqrt(np.sum((a-b)**2))


class TriangleSurvey:
    def __init__(self):
        self.triangles = []   # List of triangle point labels
        self.points = {}      # Hash of point label locations
        self.constraints = {} # Hash of point-point distances

    def constraint_label(self, a, b):
        pair = [a, b]
        pair.sort()
        return tuple(pair)

    def load(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            f.close()

        for line in lines:
            if re.match('^\s*#', line):
                # Commented line
                continue
            if re.match('^\s*$', line):
                # Blank line
                continue
            if re.search('</control>', line):
                # Don't do anything after the end of the control. 
                break

            group = line.strip().split()
            print group

            if len(group) == 3:
                self.triangles.append(group)

            if len(group) == 4:
                assert(group[2] == '=')
                pair = self.constraint_label(group[0], group[1])
                print pair
                length = float(group[3])
                self.constraints[pair] = (length, 0.03 * length + 0.01)

    def get_constraint(self, label1, label2):
        pair = self.constraint_label(label1, label2)
        return self.constraints[pair]

    def add_triangle(self, labels):
        triple = deque(labels)
        sides = [0, 0, 0]
        sides[0] = self.get_constraint(labels[0], labels[1])[0]
        sides[1] = self.get_constraint(labels[1], labels[2])[0]
        sides[2] = self.get_constraint(labels[2], labels[0])[0]
        sides = deque(sides)

        if triple[2] in self.points:
            triple.rotate(1)
            sides.rotate(1)
        if triple[2] in self.points:
            triple.rotate(1)
            sides.rotate(1)

        if triple[2] in self.points:
            # If all three points are already solved, don't do anything else.
            print 'All points already solved: ', labels
            return 

        print 'Adding point: ', tuple(triple), tuple(sides)

        self.points[triple[2]] = solve_triangle(
            self.points[triple[0]],
            self.points[triple[1]],
            sides[1],
            sides[2])

    def solve_triangles(self):
        # Clear all point locations
        self.points = {}

        # Initialize the first two points.
        t = self.triangles[0]
        # First point of the first triangle is at the origin
        self.points[t[0]] = np.array([0, 0])
        # Second point of the first triangle is at [0, d] 
        pair = self.constraint_label(t[0], t[1])
        d = self.constraints[pair][0]
        self.points[t[1]] = np.array([0, d])

        for t in self.triangles:
            self.add_triangle(t)

    def plot_triangles(self, color='b'):
        for t in self.triangles:
            p0 = self.points[t[0]]
            p1 = self.points[t[1]]
            p2 = self.points[t[2]]
            xy = np.vstack((p0, p1, p2, p0))
            mpl.plot(xy[:,0], xy[:,1], color)
            dx = 5
            dy = 5
            mpl.text(p0[0]+dx, p0[1]+dy, t[0])
            mpl.text(p1[0]+dx, p1[1]+dy, t[1])
            mpl.text(p2[0]+dx, p2[1]+dy, t[2])

    def calculate_closure(self, point_list):
        total = 0
        for i in range(len(point_list)-1):
            a = self.points[point_list[i]]
            b = self.points[point_list[i+1]]
            total += dist(a, b)
            
        a = self.points[point_list[0]]
        b = self.points[point_list[-1]]
        error = dist(a, b)

        return error, total

    def print_stats(self):
        print '%d triangles' % len(self.triangles)
        print '%d distances' % len(self.constraints)
        print '%d points' % len(self.points)

    def identify_points(self):
        def relabel(p):
            return re.sub('_close.*', '', p)

        self.new_constraints = {}

        keys = self.constraints.keys()
        for key in keys:
            k0 = relabel(key[0])
            k1 = relabel(key[1])
            key2 = self.constraint_label(k0, k1)
            self.new_constraints[key2] = self.constraints[key]

        self.new_triangles = [(relabel(t[0]), relabel(t[1]), relabel(t[2])) for t in self.triangles]

    def optimize_triangles(self):
        #point_labels = self.points.keys()
        #point_labels.sort()

        con_labels = self.new_constraints.keys()
        con_labels.sort()

        self.triangles = self.new_triangles
        
        point_labels = set()
        for c in con_labels:
            point_labels.add(c[0])
            point_labels.add(c[1])
        point_labels = list(point_labels)
        point_labels.sort()

        x0 = np.zeros(2*len(point_labels) - 2, dtype='double')
        i = 0
        # skip the first two points
        for p in point_labels[1:]:
            x0[i] = self.points[p][0]
            x0[i+1] = self.points[p][1]
            i += 2

        def x2points(x):
            i = 0
            points = {}
            points[point_labels[0]] = self.points[point_labels[0]]
            #points[point_labels[1]] = self.points[point_labels[1]]
            for p in point_labels[1:]:
                points[p] = x[i:i+2]
                i += 2
            return points
            
        def func(x): 
            points = x2points(x)
            
            dy = np.zeros(len(con_labels), dtype='double')
            for i, k in enumerate(con_labels):
                a = points[k[0]]
                b = points[k[1]]
                d1 = dist(a, b)
                con = self.new_constraints[k]
                dy[i] = (con[0] - d1) / con[1]

            print np.sum(dy**2)

            return dy 
        
        result = scipy.optimize.leastsq(func, x0)    
        print result

        x = result[0]
        self.points = x2points(x)

        return result

    def write_points(self, filename='points.dat'):
        keys = self.points.keys()
        keys.sort()
        
        with open(filename, 'w') as f:
            for k in keys:
                p = self.points[k]
                f.write('%15.6f %15.6f %s\n' % (p[0], p[1], k))

def foo():
    ts = TriangleSurvey()
    #ts.load('survey1.dat')
    ts.load('survey2.dat')
    ts.solve_triangles()
    ts.print_stats()

    print ts.triangles
    print ts.points

    inner = ['A', 'C', 'F', 'I', 'L', 'O', 'A_close']
    outer = ['B', 'D', 'E', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'B_close']
    print ts.calculate_closure(inner)
    print ts.calculate_closure(outer)

    mpl.figure(1)
    ts.identify_points()
    ts.plot_triangles(color='r')
    mpl.axes().set_aspect('equal', 'datalim')
    mpl.xlabel('Easting (feet)')
    mpl.ylabel('Northing (feet)')
    mpl.savefig('survey2_uncorrected.png')

    ts.write_points('survey2_points.dat')

    res = ts.optimize_triangles()
    ts.write_points('survey2_points_optimized.dat')
    
    ts.plot_triangles(color='k')
    mpl.axes().set_aspect('equal', 'datalim')
    mpl.savefig('survey2_both.png')

    mpl.figure(2)
    ts.plot_triangles(color='k')
    mpl.axes().set_aspect('equal', 'datalim')
    mpl.xlabel('Easting (feet)')
    mpl.ylabel('Northing (feet)')
    mpl.savefig('survey2_corrected.png')

    mpl.show()


def foo2():
    # Replace all the 100 foot measurements with measurements with 3% random
    # error on them.
    with open('survey1.dat') as f:
        lines = f.readlines()
        f.close()

    with open('survey2.dat', 'w') as f:
        for line in lines:
            num = '%.3f' % (100 + np.random.randn(1) * 3)
            line = re.sub('100', num, line)
            f.write(line)
        f.close()


if __name__ == "__main__":
    foo()
    #foo2()




