import matplotlib.pyplot as mpl
import numpy as np
import scipy.optimize

import triangle_survey as ts

class TriangleError:
  def __init__(self):
    self.triangles = ts.TriangleSurvey()

  def load(self, triangles):
    self.triangles = triangles
    point_names = list(self.triangles.points.keys())
    point_names.sort()
    self.name_to_index = {name: i for i, name in enumerate(point_names)}
    self.index_to_name = {i: name for i, name in enumerate(point_names)}
    self.constraints = {}
    for key, value in self.triangles.constraints.items():
      indices = [self.name_to_index[k] for k in key]
      indices.sort()
      pair = tuple(indices)
      self.constraints[pair] = value
    self.x = np.zeros(len(point_names) * 2, dtype='double')
    for name in point_names:
      index = self.name_to_index[name]
      print(index, name)
      self.x[2*index:2*index+2] = self.triangles.points[name]

  def chi(self, x):
    deltas = np.zeros(3 + len(self.constraints), dtype='double')
    deltas[0] = np.sum(x[::2]) * 1000
    deltas[1] = np.sum(x[1::2]) * 1000
    deltas[2] = np.sum(x[0:-1:2] * x[1::2])
    #deltas[-1] = x[-1] * 1e3
    i = 3
    for key, value in self.constraints.items():
      index0 = key[0]
      index1 = key[1]
      p0 = x[2*index0:2*index0+2]
      p1 = x[2*index1:2*index1+2]
      deltas[i] = (ts.dist(p0, p1) - value[0]) / value[1]
      i += 1
    return deltas

  def chi2(self, x):
    return np.sum(self.chi(x)**2)

  def optimize(self):
    x0 = self.x.ravel()
    x, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(self.chi, x0, full_output=True)
    self.x = x
    self.cov_x = cov_x
    #print(x)
    #print(cov_x)
    #print(infodict)
    #print(mesg)
    #print(ier)

  def debug(self):
    print(self.triangles.triangles)
    print(self.triangles.constraints)
    print(self.triangles.points)
    print()
    print('constraints = ', self.constraints)
    print('name_to_index = ', self.name_to_index)
    print('x = ', self.x)
    deltas = self.chi(self.x)
    print(len(self.x), len(deltas))
    print(deltas)
    print(self.chi2(self.x))

  def plot_constraints(self, x, color='C0', ax=None):
    for k0, k1 in self.constraints.keys():
      xx = (x[2*k0], x[2*k1])
      yy = (x[2*k0+1], x[2*k1+1])
      if ax is None:
        mpl.plot(xx, yy, color=color, alpha=0.5)
      else:
        ax.plot(xx, yy, color=color, alpha=0.5)

  def plot_error(self, scale=1):
    fig, ax = mpl.subplots(2,2)
    w, v = np.linalg.eig(self.cov_x)
    print('w = ', w)
    #print(v)
    index = 1
    for row in range(2):
      for col in range(2):
        self.plot_constraints(self.x + scale * w[index] * v[:,index], color='r', ax=ax[row][col])
        self.plot_constraints(self.x - scale * w[index] * v[:,index], color='b', ax=ax[row][col])
        self.plot_constraints(self.x, color='k', ax=ax[row][col])
        ax[row][col].set_aspect('equal')
        ax[row][col].set_title('w = %f' % w[index])
        index += 1
    mpl.show()

def foo():
  filename = 'survey3.dat'
  triangles = ts.TriangleSurvey()
  triangles.load(filename)
  triangles.solve_triangles()
  #triangles.optimize_triangles()

  te = TriangleError()
  te.load(triangles)
  te.debug()
  te.optimize()


def foo2():
  triangles = ts.TriangleSurvey()
  triangles.load('survey3.dat')
  triangles.solve_triangles()

  te = TriangleError()
  te.load(triangles)
  te.debug()
  te.optimize()
  te.plot_error(scale=1000)


def foo10():
  def chi(x):
    return x * np.array((0.9,2,3,4), dtype='double')

  x0 = np.array((1,3,0,1), dtype='double')
  print(x0)
  print(chi(x0))
  print()
  x, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(chi, x0, full_output=True)
  print(x)
  print(cov_x)
  print(infodict)
  print(mesg)
  print(ier)

if __name__ == "__main__":
  #foo()
  foo2()
  #foo2()

