import matplotlib.pyplot as plt
import numpy as np
import decimal

rr = np.arange(-10, 10, 0.2)
x = 7
learn_rate = 0.00005
xx = []
m = 2
neur = 10

def f(x):
    return x**5 - 8*(x**3) + 10*x + 6

def derivative(x):
    return 5 * x**4 - 24 * x**2 + 10

def sigma():
    res = 0
    for i in range(neur):
        res += derivative(i)
    return res

for i in range (100):
    xx.append(x)
    x = x - (learn_rate/m) * derivative(x)

xx = np.array(xx)
plt.plot(rr, f(rr))
plt.plot(xx, f(xx), 'bo')
plt.show()