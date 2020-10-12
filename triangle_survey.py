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


def angle_between(p1, p2, p3, p4):
    """ 
    p1 --------- p2
    p3 
      \
       \
        \
         \
          p4


    Provides the rotation angle in degrees needed to rotate the vector from 
    p1 to p2 to the vector from p3 to p4.
    """
    vec1 = p2 - p1
    vec2 = p4 - p3

    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])

    return (angle2 - angle1) * 180 / np.pi


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

    print('adding: ', triple, sides)

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
        self.current_fractional_error = 0.03
        self.current_absolute_error = 0.01

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
            if re.search('^FRACTIONAL ERROR', line):
                # Set the current fractional error
                group = line.strip().split()
                self.current_fractional_error = float(group[3])
                continue 
            if re.search('^ABSOLUTE ERROR', line):
                # Set the current absolute error
                group = line.strip().split()
                self.current_absolute_error = float(group[3])
                continue 

            group = line.strip().split()
            print(group)

            if len(group) == 3:
                self.triangles.append(tuple(group))

            if len(group) == 4:
                assert(group[2] == '=')
                pair = self.constraint_label(group[0], group[1])
                print(pair)
                length = float(group[3])
                self.constraints[pair] = (length, 
                    self.current_fractional_error * length + self.current_absolute_error)

            if len(group) == 5:
                assert(group[2] == '=')
                pair = self.constraint_label(group[0], group[1])
                print(pair)
                length = float(group[3])
                error = float(group[4])
                self.constraints[pair] = (length, error)

            #if len(group) == 5:
            #    self.triangles.append(group[0:3])
            #    pair = self.constraint_label(group[1], group[2])
            #    self.constraints[pair] = (length, 
            #        self.current_fractional_error * length + self.current_absolute_error)

    def save(self, filename):
        with open(filename, 'w') as f:
            for t in self.triangles:
                f.write('%s %s %s\n' % t)
            f.write('\n')
            keys = list(self.constraints.keys())
            keys.sort()
            for k in keys:
                print(k)
                c = self.constraints[k]
                f.write('%s %s = %g %g\n' % (k[0], k[1], c[0], c[1]))
            f.close()

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
            print('All points already solved: ', labels)
            return 

        print('Adding point: ', tuple(triple), tuple(sides))

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

    def plot_triangles(self, color='b', dx=5, dy=5):
        for c in self.constraints: 
            p0 = self.points[c[0]]
            p1 = self.points[c[1]]
            xy = np.vstack((p0, p1))
            if self.constraints[c][1] > 0.2:
                mpl.plot(xy[:,0], xy[:,1], color=(0.5,0.5,0.5), linestyle=':')
            else:
                mpl.plot(xy[:,0], xy[:,1], color=(0.5,0.5,0.5))

        for t in self.triangles:
            p0 = self.points[t[0]]
            p1 = self.points[t[1]]
            p2 = self.points[t[2]]
            xy = np.vstack((p0, p1, p2, p0))
            mpl.plot(xy[:,0], xy[:,1], color, linewidth=0.5)
            mpl.text(p0[0]+dx, p0[1]+dy, t[0])
            mpl.text(p1[0]+dx, p1[1]+dy, t[1])
            mpl.text(p2[0]+dx, p2[1]+dy, t[2])

    def plot_points(self, color='+b', dx=5, dy=5):
        point_set = set() 
        for t in self.triangles:
            for i in range(3):
                point_set.add(t[i])

        for p in point_set:
            p0 = self.points[p]
            mpl.plot(p0[0], p0[1], color)
            mpl.text(p0[0]+dx, p0[1]+dy, p)

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
        print('%d triangles' % len(self.triangles))
        print('%d distances' % len(self.constraints))
        print('%d points' % len(self.points))

    def identify_points(self):
        def relabel(p):
            return re.sub('_close.*', '', p)

        new_constraints = {}

        keys = self.constraints.keys()
        for key in keys:
            k0 = relabel(key[0])
            k1 = relabel(key[1])
            key2 = self.constraint_label(k0, k1)
            new_constraints[key2] = self.constraints[key]

        self.constraints = new_constraints
        self.triangles = [(relabel(t[0]), relabel(t[1]), relabel(t[2])) for t in self.triangles]

    def transform_affine(self, fixed):
        """
        Transforms all points so that the fixed points are close to their
        desired locations.
        fixed = [('name', x1, y1), ('name2', x2, y2)]
        """
        if len(fixed) == 2:
            label0 = fixed[0][0]
            label1 = fixed[1][0]

            p1 = self.points[label0]
            p2 = self.points[label1]
            p3 = np.array(fixed[0][1:3])
            p4 = np.array(fixed[1][1:3])

            theta = angle_between(p1, p2, p3, p4) * np.pi / 180

            scale = dist(p3, p4) / dist(p1, p2)
            s = np.sin(theta)
            c = np.cos(theta)
            rot = np.array([[c, -s],
                            [s, c]]) * scale

            labels = self.points.keys()
            for label in labels:
                xy = self.points[label]
                xy2 = np.dot(rot, xy - p1) + p3
                self.points[label] = xy2

        elif len(fixed) > 2:
            mat = np.zeros((2*len(fixed), 6))
            vec = np.zeros(2*len(fixed))
            for i, f in enumerate(fixed):
                label = f[0]
                mat[2*i,0] = self.points[label][0]
                mat[2*i,1] = self.points[label][1] 
                mat[2*i+1,2] = self.points[label][0] 
                mat[2*i+1,3] = self.points[label][1] 
                mat[2*i,4] = 1 
                mat[2*i+1,5] = 1 

                vec[2*i] = f[1]
                vec[2*i+1] = f[2]

            coeff, resid, rank, s = np.linalg.lstsq(mat, vec)
            a, b, c, d, e, f = tuple(coeff)

            labels = self.points.keys()
            for label in labels:
                x = self.points[label][0]
                y = self.points[label][1]

                x2 = a * x + b * y + e
                y2 = c * x + d * y + f
                self.points[label][0] = x2
                self.points[label][1] = y2

    def optimize_triangles(self, fixed=None):
        #point_labels = self.points.keys()
        #point_labels.sort()

        con_labels = list(self.constraints.keys())
        con_labels.sort()
        print('constraint labels = ', con_labels)
        print('self.triangles = ', self.triangles)
        
        point_labels = set()
        for c in con_labels:
            point_labels.add(c[0])
            point_labels.add(c[1])

        dof_subtract = 1
        if fixed:
            dof_subtract = 0
            for fix in fixed:
                point_labels.remove(fix[0])

        point_labels = list(point_labels)
        point_labels.sort()

        print('%d points = %d degrees of freedom' % (len(point_labels), 2 * len(point_labels)))
        print('%d points = %d degrees of freedom' % (len(point_labels) - dof_subtract,
            2 * len(point_labels) - 2 * dof_subtract))

        x0 = np.zeros(2*(len(point_labels) - dof_subtract), dtype='double')
        i = 0
        # skip the first point
        for p in point_labels[dof_subtract:]:
            x0[i] = self.points[p][0]
            x0[i+1] = self.points[p][1]
            i += 2

        def x_to_points(x):
            i = 0
            points = {}
            points[point_labels[0]] = self.points[point_labels[0]]
            if fixed:
                for fix in fixed:
                    points[fix[0]] = np.array((fix[1], fix[2]))
            for p in point_labels[dof_subtract:]:
                points[p] = x[i:i+2]
                i += 2
            return points
            
        # Function to optimize
        def func(x): 
            points = x_to_points(x)

            dy = np.zeros(len(con_labels), dtype='double')
            for i, k in enumerate(con_labels):
                a = points[k[0]]
                b = points[k[1]]
                d1 = dist(a, b)
                con = self.constraints[k]
                dy[i] = (con[0] - d1) / con[1]

            print(np.sum(dy**2))

            return dy 

        result = scipy.optimize.leastsq(func, x0)    
        print(result)

        x = result[0]
        self.points = x_to_points(x)

        return result

    def write_points(self, filename='points.dat'):
        keys = list(self.points.keys())
        keys.sort()

        with open(filename, 'w') as f:
            for k in keys:
                p = self.points[k]
                f.write('%15.6f %15.6f %s\n' % (p[0], p[1], k))

    #def write_constraints(self, filename='constraints.dat'):
    #    with open(filename, 'w') as f:


def foo():
    ts = TriangleSurvey()
    tag = 'survey2'
    ts.load(tag+'.dat')
    ts.solve_triangles()
    ts.print_stats()

    print(ts.triangles)
    print(ts.points)

    #inner = ['A', 'C', 'F', 'I', 'L', 'O', 'A_close']
    #outer = ['B', 'D', 'E', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'B_close']
    #print ts.calculate_closure(inner)
    #print ts.calculate_closure(outer)

    mpl.figure(1)
    ts.plot_triangles(color='r')
    mpl.axes().set_aspect('equal', 'datalim')
    mpl.xlabel('Easting (feet)')
    mpl.ylabel('Northing (feet)')
    mpl.savefig(tag+'_uncorrected.png')

    ts.identify_points()
    ts.write_points(tag+'_points.dat')

    res = ts.optimize_triangles()
    ts.write_points(tag+'_points_optimized.dat')
    
    ts.plot_triangles(color='k')
    mpl.axes().set_aspect('equal', 'datalim')
    mpl.savefig(tag+'_both.png')

    mpl.figure(2)
    ts.plot_triangles(color='k', dx=1, dy=1)
    mpl.axes().set_aspect('equal', 'datalim')
    mpl.xlabel('Easting (feet)')
    mpl.ylabel('Northing (feet)')
    mpl.tight_layout()
    mpl.savefig(tag+'_corrected.png', dpi=300)

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





