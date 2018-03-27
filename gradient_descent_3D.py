import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def f(x, y):
    return x ** 2 + y ** 2 + 25 * (2 * np.sin(x) ** 2)

def df_dx(x):
    return 2 * (x + 25 * np.sin(2*x))

def df_dy(y):
    return 2 * y


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-20.0, 20.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([f(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

x_ = 9.0
y_ = 9.0
xx = []
yy = []
learning_rate = 0.15

for i in range(1000):
    x_ = x_ - learning_rate*np.random.uniform(0,2) * df_dx(x_)
    y_ = y_ - learning_rate*np.random.uniform(0,2) * df_dy(y_)
    xx.append(x_)
    yy.append(y_)

xx = np.array(xx)
yy = np.array(yy)
zz = f(xx, yy)

ax.scatter(xx,yy,zz,color="k",s=20)
ax.plot_surface(X, Y, Z, alpha=0.4)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()