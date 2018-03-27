import numpy as np
import decimal
import matplotlib.pyplot as plt

rr = np.arange(-5, 5, 1)
x = 2
learn_rate = 0.1
xx = []

def f(x):
    return 2*x**2 + 1*x + 20

def derivative(x):
    return 4*x + 1

for i in range (10):
    xx.append(x)
    x = x - learn_rate * derivative(x)
    print(x)

xx = np.array(xx)
plt.plot(rr, f(rr))
plt.plot(xx, f(xx), 'bo')
plt.show()